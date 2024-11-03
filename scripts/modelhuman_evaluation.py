
import os
from scripts.utils_human_evaluation import merge_model_human_responses, human_model_comparison
import pandas as pd


# Load human annotations for 719 videos
humanannotfile = os.path.join(os.getcwd(),'human_annotations','human_annotations.csv')
df_h = pd.read_csv(humanannotfile)
videos_annot = df_h['video_name'].unique()

# Model annotated files
output_files = {}
output_files['distance'] = os.path.join(os.getcwd(),'human_annotations','000_labels_distance_720humanset.csv')
output_files['arousal'] = os.path.join(os.getcwd(),'human_annotations','000_labels_arousal_720humanset.csv') 
output_files['communicating'] = os.path.join(os.getcwd(),'human_annotations','000_labels_communicating_720humanset.csv')
output_files['expanse'] = os.path.join(os.getcwd(),'human_annotations','000_labels_expanse_720humanset.csv')
output_files['facingness'] = os.path.join(os.getcwd(),'human_annotations','000_labels_facingness_720humanset.csv')
output_files['joint'] = os.path.join(os.getcwd(),'human_annotations','000_labels_joint_720humanset.csv')
output_files['object'] = os.path.join(os.getcwd(),'human_annotations','000_labels_object_720humanset.csv')
output_files['valence'] = os.path.join(os.getcwd(),'human_annotations','000_labels_valence_720humanset.csv')
output_files['scene_analysis'] = os.path.join(os.getcwd(),'human_annotations','000_labels_scene_analysis_720humanset.csv')

question_type = 'distance'
output_csv_file = output_files[question_type]

# Loop to add human scores and compare
for question_type, output_csv_file in output_files.items():
    print(question_type)
    df = merge_model_human_responses(output_csv_file, humanannotfile)

# Loop to compute correlations
for question_type, output_csv_file in output_files.items():
    print(question_type)
    results = human_model_comparison(output_csv_file)


"""
distance
Question: distance
Model to Mean Human Correlation: 0.8447019731910707
Model to Human Correlations - Mean: 0.691124271315884, Std Dev: 0.031956430857087346
Human to Human Correlations - Mean: 0.6358101401490897, Std Dev: 0.04957417347535101
arousal
Question: arousal
Model to Mean Human Correlation: 0.7003007596202792
Model to Human Correlations - Mean: 0.4384431140029609, Std Dev: 0.04213129580636216
Human to Human Correlations - Mean: 0.3281711395047151, Std Dev: 0.06712716968651822
communicating
Question: communicating
Model to Mean Human Correlation: -0.6702713645749343
Model to Human Correlations - Mean: -0.4499628095592126, Std Dev: 0.02250777850748215
Human to Human Correlations - Mean: 0.3915118800702653, Std Dev: 0.052139728487949254
expanse
Question: expanse
Model to Mean Human Correlation: 0.6855943746524981
Model to Human Correlations - Mean: 0.44667397630053934, Std Dev: 0.02953617898448269
Human to Human Correlations - Mean: 0.3689122412890927, Std Dev: 0.061394028510284485
facingness
Question: facingness
Model to Mean Human Correlation: 0.8250461786659267
Model to Human Correlations - Mean: 0.739751455255756, Std Dev: 0.019782070298769352
Human to Human Correlations - Mean: 0.7916418263958211, Std Dev: 0.028400513503758076
joint
Question: joint
Model to Mean Human Correlation: 0.6931625486110463
Model to Human Correlations - Mean: 0.4986177899687706, Std Dev: 0.03078545260497225
Human to Human Correlations - Mean: 0.46714067327162795, Std Dev: 0.03540879905458441
object
Question: object
Model to Mean Human Correlation: 0.7867715026928958
Model to Human Correlations - Mean: 0.711323816106956, Std Dev: 0.019105872923607402
Human to Human Correlations - Mean: 0.7428842060993366, Std Dev: 0.02713492579491519
valence
Question: valence
Model to Mean Human Correlation: 0.6644875571590922
Model to Human Correlations - Mean: 0.41048159398842293, Std Dev: 0.04189605560413702
Human to Human Correlations - Mean: 0.30059686196214624, Std Dev: 0.06510046009924968
"""






