import cv2
import numpy
import shapely

def tissue_detection(
    image: numpy.ndarray,
    min_region_size: int = 5_000,
    max_hole_size: int = 1_500,
    outer_contours_only: bool = False,
    blur_kernel: int = 17,
    morph_kernel_size: int = 7,
    morph_n_iterations: int = 3,
    manual_threshold: int = None,
    return_poly: bool = False,
) -> numpy.ndarray:
    """For a histology image, threshold the foreground and background.

    The steps for this are as follows:
    1. Blur the image
    2. Threshold the image
    3. Performing opening and closing morphological operations to tidy up
    the initial threshold mask.
    4. Detect contours for all foreground objects in the threshold mask
    5. Discard contours that are below a threshold.
    6. Draw polygons with a value of 255

    Args:
        image (numpy.ndarray): RGB histology image to threshold.
        min_region_size (int, optional): Minimum size of foreground objects. Measured
        as total number of pixels. Defaults to 5_000.
        max_hole_size (int, optional): Maximum size threshold for hole objects to keep. Otherwise, holes 
        will be filled. Hole has to be within a valid parent contour. Defaults to 1_500.
        outer_contours_only (bool, optional): Only retain the outermost contour and fill all holes within
        this contour. Defaults to False.
        blur_kernel (int, optional): Blur kernel applied before thresholding. Must be odd. Defaults to 17.
        morph_kernel_size (int, optional): Opening and closing kernel size. Defaults to 7.
        morph_n_iterations (int, optional): Number of iterations to perform opening and closing, each. 
        Defaults to 3.
        manual_threshold (int, optional): Manual threshold to apply, rather than using Otsu. Defaults to None.
        return_poly (bool, optional): If True, returns polygon coordinates. If True, outer_contours_only must
        also be True. This is due to desiring one contour per detected object. If outer_contours_only was False,
        there would be multiple polygons returned for a shape with holes in it (below the max_hole_size threshold)

    Returns:
        numpy.ndarray: Thresholded histology image.
    """

    if return_poly:
        assert outer_contours_only, "If return_poly is True outer_contours_only must also be True."

    # Convert the image to HSV and extract the saturation channel
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation_channel = hsv_image[..., 1]

    # Apply median blur
    blurred_image = cv2.medianBlur(saturation_channel, ksize=blur_kernel)

    # Choose thresholding method
    # If no manual threshold is requested, we will use Otsu
    if manual_threshold is not None:
        threshold_type = cv2.THRESH_BINARY
    else:
        threshold_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        manual_threshold = 0

    # Apply thresholding to create a binary mask
    _, mask = cv2.threshold(
        src=blurred_image, 
        thresh=manual_threshold, 
        maxval=255, 
        type=threshold_type,
    )

    # Create the structuring element for morphological operations
    struct_elem = numpy.ones((morph_kernel_size, morph_kernel_size), dtype=numpy.uint8)

    # Perform morphological opening and closing to clean up the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct_elem, iterations=morph_n_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct_elem, iterations=morph_n_iterations)

    # Set contour retrieval mode
    mode = cv2.RETR_EXTERNAL if outer_contours_only else cv2.RETR_CCOMP

    # Find contours
    contours, hierarchy = cv2.findContours(mask.copy(), mode=mode, method=cv2.CHAIN_APPROX_NONE)

    # If no contours are found, return an empty mask
    if hierarchy is None:
        return numpy.zeros_like(mask)

    # Initialize output masks
    if not return_poly:
        # Output will be a mask array
        mask_out = numpy.zeros_like(mask, dtype=numpy.uint8)
    else:
        # Output will be a list of polygon coordinates
        mask_out = []

    if outer_contours_only:
        # Process outer contours only
        for contour in contours:
            if cv2.contourArea(contour) > min_region_size:
                if not return_poly:
                    cv2.fillPoly(mask_out, [contour], 255)
                else:
                    polggon = shapely.Polygon(contour.squeeze(axis=1))
                    mask_out.append(polggon)
    else:
        # Process both outer and inner contours
        hierarchy = numpy.squeeze(hierarchy, axis=0)

        # Identify outer and hole contours
        is_outer_contour = hierarchy[:, 3] == -1
        is_hole_contour = ~is_outer_contour

        # Create boolean masks for size thresholds
        # Only contours above the given threshold will be kept
        outer_contour_valid = numpy.array([cv2.contourArea(c) > min_region_size for c in contours])
        hole_contour_valid = numpy.array([cv2.contourArea(c) > max_hole_size for c in contours])

        # Check if hole contours have valid parents
        valid_hole_parents = numpy.array([
            hierarchy[i, 3] in numpy.where(outer_contour_valid)[0]
            for i in range(len(contours))
        ])

        # Fill the appropriate contours
        for i, contour in enumerate(contours):
            if is_outer_contour[i] and outer_contour_valid[i]:
                # Fill in contours that are above the size threshold
                cv2.fillPoly(mask_out, [contour], 255)
            # Only keep hole contours if they are valid and are within a parent
            elif is_hole_contour[i] and hole_contour_valid[i] and valid_hole_parents[i]:
                # Subtract hole from mask
                cv2.fillPoly(mask_out, [contour], 0)  

    return mask_out


