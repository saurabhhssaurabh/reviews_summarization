from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
import torch, json, logging
logging.getLogger().setLevel(logging.INFO)

class OpinionExtraction():
    model_dir = '/home/dev01/saurabh/token_classifier_r/output/bert/version_2/'
    input_file = "/home/dev01/saurabh/token_classifier_r/predict_data/input/b0013frnkg.json"
    output_file = "/home/dev01/saurabh/token_classifier_r/predict_data/output/b0013frnkg.json"
    question_list = ["how is display?", "how is memory?", "how is quality of speaker?", 
                    "how is sound?", "how is processor?", "how is wireless connection?", 
                    "how is battery?", "how is brand?", "how is operating system?",
                    "how is camera?"]

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, return_token_type_ids = True)
        self.model = BertForQuestionAnswering.from_pretrained(self.model_dir)
        self.nlp = pipeline('question-answering', model = self.model, tokenizer = self.tokenizer)

    def predict(self):
        counter = 0
        with open(self.input_file, "r") as in_f:
            with open(self.output_file, "w") as out_f:
                for line in in_f:
                    counter+=1
                    obj = json.loads(line)
                    text = obj["review"]
                    len_ = len(text)
                    start_index = 0
                    end_index = 512
                    list_1 = []
                    list_2 = []

                    while start_index < len_:
                        context = text[start_index: end_index]
                        for question in self.question_list:
                            try: 
                                output = self.nlp({'question': question, 'context': context})
                                list_1.append({"question": question, "answer": output})
                            except:
                                pass

                        list_2.append({"context": context, "qna": list_1})
                        start_index = end_index + 1
                        end_index = end_index + 512

                    obj["qna"] = list_2
                    out_f.write(json.dumps(obj)+"\n")

                    logging.info("object number => {}".format(counter))