
import os
from scripts.utils_human_evaluation import merge_model_human_responses, human_model_comparison
import pandas as pd


# Load human annotations for 719 videos
humanannotfile = os.path.join(os.getcwd(),'human_annotations','human_annotations.csv')
df_h = pd.read_csv(humanannotfile)
videos_annot = df_h['video_name'].unique()

# Model annotated files
output_files = {}
output_files['distance'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_distance.csv')
output_files['object'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_object.csv')
output_files['expanse'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_expanse.csv')
output_files['facingness'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_facingness.csv')
output_files['communicating'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_communicating.csv')
output_files['joint'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_joint.csv')
output_files['valence'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_valence.csv')
output_files['arousal'] = os.path.join(os.getcwd(),'human_annotations','000_labels_720_arousal.csv') 

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
Model to Mean Human Correlation: 0.8404815947335237
Model to Human Correlations - Mean: 0.6887321967223524, Std Dev: 0.024844040995992082
Human to Human Correlations - Mean: 0.6358101401490897, Std Dev: 0.04957417347535101
object
Question: object
Model to Mean Human Correlation: 0.756107361425097
Model to Human Correlations - Mean: 0.6806363759849349, Std Dev: 0.02195796629337972
Human to Human Correlations - Mean: 0.7428842060993366, Std Dev: 0.02713492579491519
expanse
Question: expanse
Model to Mean Human Correlation: 0.6742470902325636
Model to Human Correlations - Mean: 0.43848525996727694, Std Dev: 0.02932835347563244
Human to Human Correlations - Mean: 0.3689122412890927, Std Dev: 0.061394028510284485
facingness
Question: facingness
Model to Mean Human Correlation: 0.7693881460027648
Model to Human Correlations - Mean: 0.6828727941237406, Std Dev: 0.026635505709251335
Human to Human Correlations - Mean: 0.7916418263958211, Std Dev: 0.028400513503758076
communicating
Question: communicating
Model to Mean Human Correlation: -0.681684426459968
Model to Human Correlations - Mean: -0.4572172871950121, Std Dev: 0.027884321810356744
Human to Human Correlations - Mean: 0.3915118800702653, Std Dev: 0.052139728487949254
joint
Question: joint
Model to Mean Human Correlation: 0.6938549134106099
Model to Human Correlations - Mean: 0.5024668045418423, Std Dev: 0.02625609091787783
Human to Human Correlations - Mean: 0.46714067327162795, Std Dev: 0.03540879905458441
valence
Question: valence
Model to Mean Human Correlation: 0.676410269018713
Model to Human Correlations - Mean: 0.4222896294048241, Std Dev: 0.03407114133159932
Human to Human Correlations - Mean: 0.30059686196214624, Std Dev: 0.06510046009924968
arousal
Question: arousal
Model to Mean Human Correlation: 0.6704214897460098
Model to Human Correlations - Mean: 0.42151696898426627, Std Dev: 0.03130053681958125
Human to Human Correlations - Mean: 0.3281711395047151, Std Dev: 0.06712716968651822
"""








