import typer
from evaluate import load

import jiwer
from whisper.normalizers import EnglishTextNormalizer

app = typer.Typer()
normalizer = EnglishTextNormalizer()

@app.command()
def main(prediction_file, reference_file, normalise: bool):
    predictions = read_file(prediction_file, normalise)
    references = read_file(reference_file, normalise)
    wer_score_clean = jiwer.wer(predictions[:2620], references[:2620])
    wer_score_other = jiwer.wer(predictions[2620:], references[2620:])
    #accuracy = (1.0 - wer_score) * 100
    print("")
    print("=" * 50)
    print(f"Normalisation: {normalise}")
    print(f"Word Error Rate Clean: {wer_score_clean}")
    print(f"Word Error Rate Other: {wer_score_other}")
    #print(f"Accuracy: {accuracy:.1f}%")
    print("=" * 50)
    print("")

# Open the file in read mode and read its contents line by line
def read_file(file_path, normalise):
    lines = []
    with open(file_path, "r") as file:
        for line in file:
            if (normalise == False):
                lines.append(line.strip())
            else:
                lines.append(normalizer(line.strip()))
    return lines

if __name__ == "__main__":
    app()


