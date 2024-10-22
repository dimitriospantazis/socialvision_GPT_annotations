# Example prompt
text1 = "Is the space indoors? (A) indoors, (B) outdoors. Answer with a single letter: A or B."
prompt = f"USER: <image>\n{text1}\nASSISTANT:"

text1 = "How large is the space where people are acting? (A) small (e.g. close up manipulation of an object), (B) medium (e.g. inside a room), (C) large, (D) very large (e.g. far space (e.g. kicking a soccer goal)). Answer with a single letter: A, B, C, or D."
prompt = f"USER: <image>\n{text1}\nASSISTANT:"

text1 = "How large is the space where people are acting?"
prompt = f"USER: <image>\n{text1}\nASSISTANT:"

#text1 = (
#    "Describe the size of the space in the scene based on the following criteria. "
#    "At the end, select a single letter: \n"
#    "- (A) small (e.g., close-up manipulation of an object)\n"
#    "- (B) medium (e.g., inside a room)\n"
#    "- (C) large (e.g., an outdoor area)\n"
#    "- (D) very large (e.g., a soccer field or far space)\n\n"
#    "Scene: [Description of your input image]\n\n"
#    "Please describe the scene briefly and provide your answer as a single letter (A, B, C, or D) at the end."
#)
#prompt = f"USER: <image>\n{text1}\nASSISTANT:"

# Expanse
text = (
    "Please analyze the following scene and answer how large the space is where people are acting. "
    "Use only one letter:\n\n"
    "(A) small (e.g., close-up manipulation of an object)\n"
    "(B) medium (e.g., inside a room)\n"
    "(C) large (e.g., an outdoor area)\n"
    "(D) very large (e.g., a soccer field or space far from the subject)\n\n"
    "Example 1:\n"
    "Scene: A person is sitting at a desk, typing on a laptop.\n"
    "Answer: B\n\n"
    "Example 2:\n"
    "Scene: A man kicking a soccer ball toward a goal from midfield.\n"
    "Answer: D\n\n"
    f"Now, please answer for the scene in the provided image"
)
prompt = f"USER: <image>\n{text}\nASSISTANT:" # Combine with the LLaVA input format
prompt_type = 'expanse'









# Object directedness - Is anyone in the video interacting with an object?
# Object Interaction Prompt
text = (
    "Please analyze the following scene and answer whether anyone is interacting with an object. "
    "Respond with only one word: 'Yes' or 'No'.\n\n"
    "Example 1:\n"
    "Scene: A person is holding a coffee mug and taking a sip.\n"
    "Answer: Yes\n\n"
    "Example 2:\n"
    "Scene: A person is standing still, looking out of a window without touching anything.\n"
    "Answer: No\n\n"
    "Example 3:\n"
    "Scene: A person is holding the hand of a child.\n"
    "Answer: No\n\n"
    f"Now, please answer for the scene in the provided image or video."
)
prompt = f"USER: <image>\n{text}\nASSISTANT:" # Combine with the LLaVA input format
prompt_type = 'object'


# Object directedness - Is anyone in the video interacting with an object?
# Object Interaction Prompt
text = (
    "Please analyze the following scene and answer whether anyone is interacting with an object."
    "Respond with only one word: 'Yes' or 'No'.\n\n"
    "Example 1:\n"
    "Scene: A person is holding a coffee mug and taking a sip.\n"
    "Answer: Yes\n\n"
    "Example 2:\n"
    "Scene: A person is standing still, looking out of a window without touching anything.\n"
    "Answer: No\n\n"
    "Example 3:\n"
    "Scene: A person is holding the hand of a child.\n"
    "Answer: No\n\n"
    f"Now, please answer for the scene in the provided image or video."
)
prompt = f"USER: <image>\n{text}\nASSISTANT:" # Combine with the LLaVA input format
prompt_type = 'object'

text = (
    "Please analyze the following scene and answer whether anyone is interacting with an inanimate object. "
    "Do not count interactions with other people, animals, or living beings as objects. "
    "If the answer is 'Yes', also name the inanimate object they are interacting with. "
    "Respond in the format: 'Yes, [object]' or 'No'.\n\n"
    
    "Example 1:\n"
    "Scene: A person is holding a coffee mug and taking a sip.\n"
    "Answer: Yes, coffee mug\n\n"
    
    "Example 2:\n"
    "Scene: A person is standing still, looking out of a window without touching anything.\n"
    "Answer: No\n\n"
    
    "Example 3:\n"
    "Scene: A person is holding the hand of a child.\n"
    "Answer: No\n\n"
    
    "Example 4:\n"
    "Scene: A person is petting a dog.\n"
    "Answer: No\n\n"

    "Example 5:\n"
    "Scene: A person is reading a book.\n"
    "Answer: Yes, book\n\n"

    f"Now, please answer for the scene in the provided image or video."
)
prompt = f"USER: <image>\n{text}\nASSISTANT:" # Combine with the LLaVA input format
prompt_type = 'object'








































# Object directedness - Is anyone in the video interacting with an object?
# Object Interaction Prompt
text = (
    "Please analyze the following scene and answer whether anyone is interacting with an object. "
    "Respond with only one word: 'Yes' or 'No'.\n\n"
    "Example 1:\n"
    "Scene: A person is holding a coffee mug and taking a sip.\n"
    "Answer: Yes\n\n"
    "Example 2:\n"
    "Scene: A person is standing still, looking out of a window without touching anything.\n"
    "Answer: No\n\n"
    f"Now, please answer for the scene in the provided image or video."
)
prompt = f"USER: <image>\n{text}\nASSISTANT:" # Combine with the LLaVA input format
prompt_type = 'object'




text = "Are the people in this image directing their attention toward an object? Answer 'yes' or 'no'."

text = "Are the people in this image directing their attention toward an object and if so what is the name of the object?"

text = "Describe the scene."
text = "In this image, do the people appear focused on an object, like a book, tool, or food item? If yes, how do they show their attention toward it?"

text = "Analyze the following image carefully. Make sure to describe the objects and actions clearly, avoiding assumptions unless explicitly visible."


text = (
    "Please analyze the following scene and answer whether anyone is interacting with an inanimate object. "
    "Do not count interactions with other people, animals, or living beings as objects. "
    "If the answer is 'Yes', also name the inanimate object they are interacting with. "
    "Respond in the format: 'Yes, [object]' or 'No'.\n\n"
    
    "Example 1:\n"
    "Scene: A person is holding a coffee mug and taking a sip.\n"
    "Answer: Yes, coffee mug\n\n"
    
    "Example 2:\n"
    "Scene: A person is standing still, looking out of a window without touching anything.\n"
    "Answer: No\n\n"
    
    "Example 3:\n"
    "Scene: A person is holding the hand of a child.\n"
    "Answer: No\n\n"
    
    "Example 4:\n"
    "Scene: A person is petting a dog.\n"
    "Answer: No\n\n"

    "Example 5:\n"
    "Scene: A person is reading a book.\n"
    "Answer: Yes, book\n\n"

    f"Now, please answer for the scene in the provided image or video."
)


text = (
    "Please analyze the following scene and answer whether anyone is directing their attention towards an object."
    "If the answer is 'Yes', also name the inanimate object they are interacting with."
    "Respond in the format: 'Yes, [object]' or 'No'.\n\n"
    f"Now, please answer for the scene in the provided image."
)

text = "Are the people in this image directing their attention toward an object? Answer 'yes' or 'no'."


text = "In this image, do the people appear focused on an object, like a book, tool, or food item? If yes, how do they show their attention toward it?"

text = "Describe the scene, focusing on "

text = "What is the object that the people are attending to in the image? Describe how their attention is directed (e.g., gaze, body orientation, hand gestures)."








# Object directedness - Is anyone in the video interacting with an object?
# Object Interaction Prompt
text = (
    "Please analyze the following scene and answer whether anyone is interacting with an object. "
    "Respond with only one word: 'Yes' or 'No'.\n\n"
    "Example 1:\n"
    "Scene: A person is holding a coffee mug and taking a sip.\n"
    "Answer: Yes\n\n"
    "Example 2:\n"
    "Scene: A person is standing still, looking out of a window without touching anything.\n"
    "Answer: No\n\n"
    f"Now, please answer for the scene in the provided image or video."
)
question_type = 'object'


text = (
    "Please analyze the following scene and answer whether anyone is interacting with an object.\n"
    "Example 1:\n"
    "Scene: A person is holding a coffee mug and taking a sip.\n"
    "Answer: Yes\n\n"
    "Example 2:\n"
    "Scene: A person is standing still, looking out of a window without touching anything.\n"
    "Answer: No\n\n"
    f"Now, please answer for the scene in the provided image or video."
)
question_type = 'object'

text = (
    "On a scale from 0 to 1, where 0 means very far and 1 means physically very close, how close are the people in this scene? "
    "Provide the score as <score>.\n\n"
    "Example 1:\n"
    "Scene: Two people are sitting side by side on a bench, their shoulders touching.\n"
    "Answer: <score> = 0.9\n\n"
    "Example 2:\n"
    "Scene: Two individuals are standing across a large room, not interacting.\n"
    "Answer: <score> = 0.1\n\n"
    f"Now, please estimate the score for the scene in the provided image or video. Answer: <score> = "
)

text = (
    "On a scale from 0 to 1, where 0 means very close and 1 means physically very far, how close are the people in this scene? "
    "Provide a single number.\n\n"
    "Example 1:\n"
    "Scene: Two people are sitting side by side on a bench, their shoulders touching.\n"
    "Answer: 0\n\n"
    "Example 2:\n"
    "Scene: Two individuals are standing across a large room, not interacting.\n"
    "Answer: 1\n\n"
    "Example 3:\n"
    "Scene: A group of friends are standing near each other, chatting closely.\n"
    "Answer: 0.2\n\n"
    f"Now, please estimate the score for the scene in the provided image. Respond with just a number."
)











text1 = (
    "First, describe the scene in detail, focusing on the people—their positions, actions, and interactions with each other. "
    "Be specific about how the people are positioned relative to one another and what they are doing."
)

text2 = (
    f"Here is the image description: '{{}}'. "
    "Based on this description, on a scale from 0 to 1, where 0 means the people are touching and 1 means they are physically very far apart, "
    "how close are the people in this scene? Use increments of 0.1 in your response and provide only the number."
)

#Mine:
text1 = ("First, describe the scene briefly, focusing on the people—how they are positioned, their actions, and their interactions. "
    "Then, describe how close the people are in this scene being as specific as possible in regards to their distance.")
text2 = f"Here is the image description: '{{}}'. Given the image description, on a scale from 0 to 1, where 0 means touching and 1 means physically very far, how close are the people in this scene? Only provide a single number, but don't only use 0 or 1 but also increments of 0.1."





