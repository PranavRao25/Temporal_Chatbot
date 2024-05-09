from imports import *

class Interface:
    def __init__(self):
        self.file_name = f"./temp_files/temp_{datetime.now()}.txt"
        self.f = open(self.file_name, 'w')
        self.f.write('Generate: ')
        self.f.close()

        self.f = open(self.file_name, 'r+')
        self.model = T5ForConditionalGeneration.from_pretrained("../savepoints_2024-04-21 20:23:55.826915/model")
        self.tokenizer = T5Tokenizer.from_pretrained("../savepoints_2024-04-21 20:23:55.826915/tokenizer")

    def closeInterface(self):
        self.f.close()
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        else:
            print("The file does not exist")

    def printTranscript(self):
        self.f.close()

        news = f"./printed_files/Conversation_{datetime.now()}.txt"
        file = open(news, 'w')
        file.close()

        shutil.copy(self.file_name, news)
        self.f = open(self.file_name, 'r+')

    def userInput(self, inputLine):
        input = inputLine + '\n'
        self.f.write(input)

        # Tokenize the text and generate output
        self.f.seek(0)
        mod_input = self.f.read() + '\n'
        inputs = self.tokenizer(mod_input, return_tensors="pt")
        outputs = self.model.generate(**inputs)

        # Decode the output tokens to text
        model_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = model_result + '\n'
        self.f.write(result)

    def displayTranscript(self):
        self.f.seek(0)
        l = self.f.readlines()
        for i in l:
            print(i)

    def clearTranscript(self):
        self.f.close()
        with open(self.file_name, 'w') as _:
            pass
        self.f = open(self.file_name, "r+")
