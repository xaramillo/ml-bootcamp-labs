from sklearn.metrics import accuracy_score, recall_score, f1_score

def evaluate_model(model,X_test,y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)
    return acc