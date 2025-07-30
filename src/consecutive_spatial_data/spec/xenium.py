from enum import Enum


class Xenium(Enum):
    pixel_size = 0.2125

    @classmethod
    def bad_transcripts(cls):
        bad_transcripts = [
            "NegControlProbe_",
            "antisense_",
            "NegControlCodeword_",
            "BLANK_",
            "Blank-",
            "NegPrb",
        ]
        return bad_transcripts
