import typer
import whisper
from evaluate import load
from whisper_norm import EnglishTextNormalizer

app = typer.Typer()
wer = load("wer")
normalizer = EnglishTextNormalizer()

@app.command()
def main(prediction_file, reference_file, normalise: bool):
    predictions = read_file(prediction_file, normalise)
    references = read_file(reference_file, normalise)
    print("Average length of predictions and references: ")
    print(average_len(predictions))
    print(average_len(references))
    wer_score = wer.compute(predictions=predictions, references=references)
    accuracy = (1.0 - wer_score) * 100
    print("")
    print("=" * 50)
    print(f"Normalisation: {normalise}")
    print(f"Word Error Rate: {wer_score}")
    print(f"Accuracy: {accuracy:.1f}%")
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

def average_len(predictions):
    model = whisper.load_model('tiny')
    tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language='en', task='transcribe')
    total_len = 0
    total_count = 0
    for prediciton in predictions:
        total_len += len(tokenizer.encode(prediciton))
        total_count += 1
    return total_len / float(total_count)

if __name__ == "__main__":
    app()


