import json

input_file = "/home/dev01/saurabh/token_classifier_r/data/bert/version_3/train.json"
output_file = "/home/dev01/saurabh/token_classifier_r/data/bert/version_3/train_squad.json"

# input_file = "/home/dev01/saurabh/token_classifier_r/data/bert/version_3/dev.json"
# output_file = "/home/dev01/saurabh/token_classifier_r/data/bert/version_3/dev_squad.json"


def rrc_to_squad():
    counter = 0
    with open(input_file, "r") as in_f:
        obj = json.load(in_f)
        paragraphs = []

        obj_counter = 0
        for obj_1 in obj["data"]:
            obj_counter = obj_counter + 1
            qas = []
            for index in range(len(obj_1["answers"])):
                if obj_1["answers"][index]["span_start"] < 0:
                    continue
                else:
                    counter = counter + 1
                    qas.append({
                        "answers": [{"answer_start": obj_1["answers"][index]["span_start"], "text": 
                        obj_1["answers"][index]["span_text"]}],
                        "question": obj_1["questions"][index]["input_text"],
                        "id": "{}".format(counter)
                    })

            paragraphs.append({
                "context": obj_1["story"],
                "qas": qas,

            })

            if obj_counter == 5:
                break

        output = {
            "data": [
                {
                    "title": "RRC dataset",
                    "paragraphs": paragraphs
                }
                ] 
        }

        with open(output_file, "w") as out_f:
            out_f.write(json.dumps(output))



if __name__ == "__main__":
    rrc_to_squad()