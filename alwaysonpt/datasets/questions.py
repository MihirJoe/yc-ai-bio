"""
Reasoning question bank for out-of-distribution evaluation.
Each dataset gets questions with ground truth derivable from dataset labels.
"""

QUESTION_BANK = {
    'gaitpdb': [
        {
            'id': 'severity',
            'task_type': 'gait_severity',
            'question': (
                'Analyze this gait recording and assess the severity of motor impairment '
                'on a scale from none (healthy) to severe. Provide your staging.'
            ),
            'answer_key': 'hoehn_yahr',
            'eval_type': 'ordinal',
            'mapping': {0: 'none', 2: 'mild', 2.5: 'mild-moderate', 3: 'moderate'},
            'extract_answer': lambda gt: gt.get('hoehn_yahr', 0) if gt.get('group') == 'patient' else 0,
        },
        {
            'id': 'healthy_vs_pd',
            'task_type': 'gait_severity',
            'question': (
                'Based on the gait force patterns, is this subject a healthy control '
                'or do they show signs of Parkinsonian gait? Explain your reasoning '
                'from the signal evidence.'
            ),
            'answer_key': 'group',
            'eval_type': 'binary',
            'positive_label': 'patient',
            'extract_answer': lambda gt: gt.get('group', 'control'),
        },
        {
            'id': 'asymmetry',
            'task_type': 'open_analysis',
            'question': (
                'Quantify the left-right asymmetry in this gait recording. '
                'Is the asymmetry clinically significant?'
            ),
            'answer_key': None,
            'eval_type': 'reasoning_quality',
            'extract_answer': lambda gt: None,
        },
    ],
    'gaitndd': [
        {
            'id': 'disease_type',
            'task_type': 'gait_diagnosis',
            'question': (
                'This gait timing data is from a patient. Based on the stride variability, '
                'swing/stance ratios, and temporal patterns, which condition is most likely: '
                'Parkinsons disease, Huntingtons disease, ALS, or healthy control?'
            ),
            'answer_key': 'disease',
            'eval_type': 'categorical',
            'categories': ['parkinsons', 'huntingtons', 'als', 'control'],
            'extract_answer': lambda gt: gt.get('disease', 'control'),
        },
        {
            'id': 'gait_stability',
            'task_type': 'open_analysis',
            'question': (
                'Assess the overall gait stability and fall risk for this subject '
                'based on stride-to-stride variability.'
            ),
            'answer_key': None,
            'eval_type': 'reasoning_quality',
            'extract_answer': lambda gt: None,
        },
    ],
    'ptb_xl': [
        {
            'id': 'ecg_diagnosis',
            'task_type': 'ecg_interpretation',
            'question': (
                'Analyze this 12-lead ECG and provide your diagnostic interpretation. '
                'What is the primary finding?'
            ),
            'answer_key': 'diagnostic_superclass',
            'eval_type': 'categorical',
            'categories': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
            'extract_answer': lambda gt: gt.get('diagnostic_superclass', 'NORM'),
        },
        {
            'id': 'ecg_normal_vs_abnormal',
            'task_type': 'ecg_interpretation',
            'question': (
                'Is this ECG normal or abnormal? If abnormal, rate the severity '
                'and urgency of the finding.'
            ),
            'answer_key': 'diagnostic_superclass',
            'eval_type': 'binary',
            'positive_label': 'abnormal',
            'extract_answer': lambda gt: 'normal' if gt.get('diagnostic_superclass') == 'NORM' else 'abnormal',
        },
    ],
    'chfdb': [
        {
            'id': 'chf_markers',
            'task_type': 'open_analysis',
            'question': (
                'Analyze this ECG segment from a cardiac patient. Identify any signs '
                'of heart failure or cardiac dysfunction from the signal morphology and rhythm.'
            ),
            'answer_key': None,
            'eval_type': 'reasoning_quality',
            'extract_answer': lambda gt: None,
        },
    ],
    'chf2db': [
        {
            'id': 'nyha_class',
            'task_type': 'hrv_analysis',
            'question': (
                'Based on the heart rate variability patterns in these RR intervals, '
                'assess the functional capacity of this heart failure patient. '
                'Estimate the NYHA functional class (I=mild, II=moderate, III=severe).'
            ),
            'answer_key': 'nyha',
            'eval_type': 'ordinal',
            'mapping': {1: 'I', 2: 'II', 3: 'III', 4: 'IV'},
            'extract_answer': lambda gt: gt.get('nyha', 2),
        },
        {
            'id': 'hrv_analysis',
            'task_type': 'hrv_analysis',
            'question': (
                'Perform a heart rate variability analysis on these RR intervals. '
                'What do the time-domain and frequency-domain HRV metrics suggest '
                'about autonomic function?'
            ),
            'answer_key': None,
            'eval_type': 'reasoning_quality',
            'extract_answer': lambda gt: None,
        },
    ],
}


def get_questions(dataset_name: str) -> list:
    """Get all questions for a dataset."""
    return QUESTION_BANK.get(dataset_name, [])


def get_question(dataset_name: str, question_id: str) -> dict:
    """Get a specific question by dataset and ID."""
    for q in QUESTION_BANK.get(dataset_name, []):
        if q['id'] == question_id:
            return q
    return None
