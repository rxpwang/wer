with open('librispeech_test_clean.txt', 'r') as file:
    lines_clean = file.readlines()

with open('text_other', 'r') as file:
    lines_other = file.readlines()

lines_all = lines_clean + lines_other

with open('librispeech_test.txt', 'w') as file:
    file.writelines(lines_all)

