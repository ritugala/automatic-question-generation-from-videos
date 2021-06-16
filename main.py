from pipelines import pipeline
from answers_similarity import similarity

def generate_text(text):
	timestamped_text = {}
	all_text = ""
	split_text = text.split('\n')

	for i in range(len(split_text) - 1):
		if i % 2 == 0:
			timestamped_text[split_text[i]] = split_text[i + 1]
		else:
			all_text += " " + split_text[i]

	all_text = all_text.lstrip()
	return timestamped_text, all_text


def answer_similarity(model_answer, student_answer):
	score = similarity(model_answer, student_answer)
	if score < 0.33:
		return "low similarity"
	elif score < 0.66:
		return "medium similarity"
	else:
		return "high similarity"

file = open("video_1_transcript.txt")
text = file.read()

timestamped_text, plain_text = generate_text(text)

nlp = pipeline("question-generation")

all_questions = nlp()

for question in all_questions:

	print("Sentence: " + question["sentence"])
	print("Question: " + question["question"])
	print("Answer: " + question["answer"])
	print("Index:" + str(question["index"]))

	# Image labels and any additional labels provided by lecturer
	if ("equation" in question["question"].lower() or "graph" in question["question"].lower()):
		print("## Question Can have image, The screenshot of the video at this time is passed to the image detection model ##")



