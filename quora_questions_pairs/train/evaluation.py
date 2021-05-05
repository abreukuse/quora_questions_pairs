from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluation(y_true, y_pred):
    """Validation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'\n*** Validation Scores ***\n'
          f'accuracy: {accuracy}\n'
          f'precision: {precision}\n'
          f'recall: {recall}\n'
          f'f1_score: {f1}\n\n')