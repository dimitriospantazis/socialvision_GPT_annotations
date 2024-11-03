# Video Annotation for Social Vision Using OpenAI's GPT API

This project provides an automated annotation pipeline for videos from the **Moments in Time** dataset, using OpenAI's GPT models to analyze and classify social interactions within each video. The goal is to generate insightful metadata regarding the social dynamics and environmental context of the scenes, which is valuable for applications in social vision research, human behavior understanding, and multimedia content analysis.

## Project Overview

### Purpose
Social interactions in videos can be subtle and complex, involving nuanced factors such as distance between people, engagement in joint actions, and environmental context. This project leverages GPT’s language model capabilities to infer and score these attributes from video frames, providing detailed annotations on various social vision dimensions, including:

1. **Proximity**: Measuring how close individuals are to each other.
2. **Object Interaction**: Determining if people are engaged with physical objects versus each other.
3. **Spatial Expanse**: Assessing the size and openness of the environment in the scene.
4. **Facingness**: Evaluating how much individuals are facing each other.
5. **Communication**: Detecting verbal or non-verbal communication.
6. **Joint Action**: Measuring the level of coordination in group activities.
7. **Valence**: Gauging the emotional tone (positive or negative) of interactions.
8. **Arousal**: Measuring the intensity of emotions or energy in the scene.
9. **Scene Analysis**: Estimating the number of people and determining if the scene is indoors or outdoors.

### Approach
The project is designed to automate video annotation by extracting key frames from each video and prompting GPT with a set of specialized questions. These prompts guide the model to assess various social and environmental characteristics, providing structured outputs with confidence scores to indicate the model's certainty for each response.

#### Sample Prompts
Each prompt is carefully constructed to capture specific aspects of social interactions, such as the "distance" between individuals, their "communication" level, or the "valence" of their interactions. Here is an example of a prompt used to assess distance:

\`\`\`python
prompt['distance'] = """
Here are several frames from a video. On a scale from 0 to 1, where 0 means physically touching and 1 means very far, how close are the people in this scene? Use the entire range, with intervals of 0.1. Provide the score as `<score> = X`.

Additionally, please provide a confidence score indicating how certain you are about this distance rating. On a scale from 0 to 1, where 0 means very low confidence and 1 means very high confidence, rate your certainty for the provided distance score. Provide the confidence score as `<confidence> = Y`.

Example 1:
Scene: Two people are sitting side by side on a bench, their shoulders touching.
Answer: <score> = 0.0
Confidence: <confidence> = 0.9

Example 2:
Scene: Two individuals are standing across a large room, not interacting.
Answer: <score> = 1.0
Confidence: <confidence> = 0.95

Now, please estimate the score and confidence level for the scene in the provided images:
Answer: <score> =
Confidence: <confidence> =
"""
\`\`\`

### How It Works
1. **Video Frame Extraction**: The pipeline extracts several equidistant frames from each video, focusing on moments that are representative of the video’s content.
2. **Prompt Generation and Inference**: For each video, specific prompts are sent to the GPT model, designed to gather responses about social interactions and spatial context. 
3. **Result Compilation**: Each annotation, including confidence scores, is compiled into a structured format, allowing for efficient data analysis and filtering.
4. **Output Storage**: The annotated results are saved in a CSV file for further analysis, enabling insights into social interactions across the Moments in Time dataset.

### Benefits
This project provides an efficient way to analyze and categorize social interactions in videos, leveraging powerful language models to understand nuanced human behaviors and environmental settings. The outputs can serve as a foundation for more extensive research or applications in social dynamics, multimedia analysis, and computer vision.

---