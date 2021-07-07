import constants as cnst

def compute_binary_confusion_matrix(confusion_matrix, emotion):
    # Because confusion matrix has rows and columns indexed from 0 to 5, but emotions are from 1 to 6
    emotion -= 1

    # Classification measures
    TP = confusion_matrix.loc[emotion, emotion]
    FP = confusion_matrix[emotion].values.sum() - TP
 
    FN = confusion_matrix.loc[emotion].values.sum() - TP
    TN = confusion_matrix.values.sum() - TP - FP - FN

    # Classification rate
    CR = (TP + TN) / (TP + TN + FP + FN)

    # Recall, precision rates and F1 measures
    recall1 = TP / (TP + FN)
    precision1 = TP / (TP + FP)
    F1 = 2 * precision1 * recall1 / (precision1 + recall1)

    # Recall, precision rates and F2 measures
    recall2 =  TN / (TN + FP)
    precision2 = TN / (TN + FN)
    F2 = 2 * precision2 * recall2 / (precision2 + recall2)

    UAR = (recall1 + recall2) / 2

    measures = {'CR': CR, 'UAR': UAR,
                'R1': recall1, 'P1': precision1, 'F1': F1,
                'R2': recall2, 'P2': precision2, 'F2': F2}

    return {cnst.EMOTIONS_LIST[emotion]: measures}