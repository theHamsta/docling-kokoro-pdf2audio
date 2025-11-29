from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import argparse
from kokoro import KPipeline
import av
import torch
import numpy as np

SAMPLE_RATE = 24000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("document", help="document to convert to audio")
    parser.add_argument("--output", "-o", help="output document", required=True)
    parser.add_argument("--md", help="Path to output markdown")
    parser.add_argument(
        "--audiocodec", "-a", help="Audio codec to use for output", default="mp3"
    )
    parser.add_argument(
        "--bitrate",
        "-b",
        help="Bitrate for output audio codec",
        default=128 * 1000,
        type=int,
    )

    args = parser.parse_args()

    source = args.document
    ###### USING SIMPLE DEFAULT VALUES
    # - GraniteDocling model
    # - Using the transformers framework

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
            ),
        }
    )

    doc = converter.convert(source=source).document

    text = doc.export_to_markdown()

    if args.md:
        with open(args.md, "w") as f:
            f.write(text)

    pipeline = KPipeline(lang_code="a")
    generator = pipeline(text, voice="af_heart")

    output_container = av.open(args.output, "w")
    output_stream = output_container.add_stream(
        args.audiocodec, rate=SAMPLE_RATE, layout="mono"
    )
    output_stream.bit_rate = args.bitrate

    chunks = []
    for i, (gs, ps, audio) in enumerate(generator):
        chunk = (audio * (2**15 - 1)).to(torch.int16).cpu().numpy()
        chunk = np.reshape(chunk, (1, -1))
        chunks.append(chunk)

    joined = np.concat(chunks, axis=-1)

    for frame_data in np.array_split(joined, SAMPLE_RATE, axis=-1):
        frame = av.AudioFrame.from_ndarray(frame_data, layout="mono")
        frame.rate = SAMPLE_RATE
        paket = output_stream.encode(frame)
        output_container.mux(paket)

    # Flush the encoder
    for out_packet in output_stream.encode():
        output_container.mux(out_packet)

    output_container.close()


if __name__ == "__main__":
    main()
