
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  

train_results = pd.read_csv("train_preds.csv")
valid_results = pd.read_csv("valid_preds.csv")

class_labels = ['Cardiomegaly',
 'Emphysema',
 'Effusion',
 'Hernia',
 'Infiltration',
 'Mass',
 'Nodule',
 'Atelectasis',
 'Pneumothorax',
 'Pleural_Thickening',
 'Pneumonia',
 'Fibrosis',
 'Edema',
 'Consolidation']

pred_labels = [l + "_pred" for l in class_labels]

y = valid_results[class_labels].values
pred = valid_results[pred_labels].values

valid_results[np.concatenate([class_labels, pred_labels])].head()

plt.xticks(rotation=90)
plt.bar(x = class_labels, height= y.sum(axis=0));

def true_positives(y, pred, th=0.5):
  
    TP = 0
    
    thresholded_preds = pred >= th

    TP = np.sum((y == 1) & (thresholded_preds == 1))
    
    return TP

def true_negatives(y, pred, th=0.5):

    TN = 0
    
    thresholded_preds = pred >= th

    TN = np.sum((y == 0) & (thresholded_preds == 0))
        
    return TN

def false_positives(y, pred, th=0.5):

    FP = 0
    
    thresholded_preds = pred >= th
    
    FP = np.sum((y == 0) & (thresholded_preds == 1))
       
    return FP

def false_negatives(y, pred, th=0.5):

    FN = 0
    
    thresholded_preds = pred >= th

    FN = np.sum((y == 1) & (thresholded_preds == 0))
       
    return FN

df = pd.DataFrame({'y_test': [1,1,0,0,0,0,0,0,0,1,1,1,1,1],
                   'preds_test': [0.8,0.7,0.4,0.3,0.2,0.5,0.6,0.7,0.8,0.1,0.2,0.3,0.4,0],
                   'category': ['TP','TP','TN','TN','TN','FP','FP','FP','FP','FN','FN','FN','FN','FN']
                  })

display(df)

y_test = df['y_test']


preds_test = df['preds_test']

threshold = 0.5
print(f"threshold: {threshold}\n")

print(f"""Our functions calcualted: 
TP: {true_positives(y_test, preds_test, threshold)}
TN: {true_negatives(y_test, preds_test, threshold)}
FP: {false_positives(y_test, preds_test, threshold)}
FN: {false_negatives(y_test, preds_test, threshold)}
""")

print("Expected results")
print(f"There are {sum(df['category'] == 'TP')} TP")
print(f"There are {sum(df['category'] == 'TN')} TN")
print(f"There are {sum(df['category'] == 'FP')} FP")
print(f"There are {sum(df['category'] == 'FN')} FN")

util.get_performance_metrics(y, pred, class_labels)

def get_accuracy(y, pred, th=0.5):
 
    accuracy = 0.0
    
functions
    TP = true_positives(y, pred, th=th)
    FP = false_positives(y, pred, th=th)
    TN = true_negatives(y, pred, th=th)
    FN = false_negatives(y, pred, th=th)

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    
    return accuracy

print("Test case:")

y_test = np.array([1, 0, 0, 1, 1])
print('test labels: {y_test}')

preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
print(f'test predictions: {preds_test}')

threshold = 0.5
print(f"threshold: {threshold}")

print(f"computed accuracy: {get_accuracy(y_test, preds_test, threshold)}")

util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy)


def get_prevalence(y):

    prevalence = 0.0
    
    prevalence = np.mean(y)
      
    return prevalence

print("Test case:\n")

y_test = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
print(f'test labels: {y_test}')

print(f"computed prevalence: {get_prevalence(y_test)}")
Test case:

test labels: [1 0 0 1 1 0 0 0 0 1]
computed prevalence: 0.4
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence)

def get_sensitivity(y, pred, th=0.5):

    sensitivity = 0.0
 
    
    TP = true_positives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)

    sensitivity = TP / (TP + FN)
      
    return sensitivity

def get_specificity(y, pred, th=0.5):

    specificity = 0.0
    
    TN = true_negatives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)
    
    specificity = TN / (TN + FP)
    
    return specificity

print("Test case")

y_test = np.array([1, 0, 0, 1, 1])
print(f'test labels: {y_test}\n')

preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
print(f'test predictions: {preds_test}\n')

threshold = 0.5
print(f"threshold: {threshold}\n")

print(f"computed sensitivity: {get_sensitivity(y_test, preds_test, threshold):.2f}")
print(f"computed specificity: {get_specificity(y_test, preds_test, threshold):.2f}")

util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity)

def get_ppv(y, pred, th=0.5):

    PPV = 0.0
  
    TP = true_positives(y, pred, th=0.5)
    FP = false_positives(y, pred, th=0.5)

    PPV = TP / (TP + FP)
 
    return PPV

def get_npv(y, pred, th=0.5):

    NPV = 0.0
     
    TN = true_negatives(y, pred, th=0.5)
    FN = false_negatives(y, pred, th=0.5)

    NPV = TN / (TN + FN)
    
    return NPV

print("Test case:\n")

y_test = np.array([1, 0, 0, 1, 1])
print(f'test labels: {y_test}')

preds_test = np.array([0.8, 0.8, 0.4, 0.6, 0.3])
print(f'test predictions: {preds_test}\n')

threshold = 0.5
print(f"threshold: {threshold}\n")

print(f"computed ppv: {get_ppv(y_test, preds_test, threshold):.2f}")
print(f"computed npv: {get_npv(y_test, preds_test, threshold):.2f}")

util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv)

util.get_curve(y, pred, class_labels)

from sklearn.metrics import roc_auc_score
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score)


def bootstrap_auc(y, pred, classes, bootstraps = 100, fold_size = 1000):
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y[:, c]
        df.loc[:, 'pred'] = pred[:, c]
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):

            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

statistics = bootstrap_auc(y, pred, class_labels)

util.print_confidence_intervals(class_labels, statistics)

from sklearn.metrics import f1_score
util.get_performance_metrics(y, pred, class_labels, acc=get_accuracy, prevalence=get_prevalence, 
                        sens=get_sensitivity, spec=get_specificity, ppv=get_ppv, npv=get_npv, auc=roc_auc_score,f1=f1_score)



from sklearn.calibration import calibration_curve
def plot_calibration_curve(y, pred):
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()

plot_calibration_curve(y, pred)

from sklearn.linear_model import LogisticRegression as LR 

y_train = train_results[class_labels].values
pred_train = train_results[pred_labels].values
pred_calibrated = np.zeros_like(pred)

for i in range(len(class_labels)):
    lr = LR(solver='liblinear', max_iter=10000)
    lr.fit(pred_train[:, i].reshape(-1, 1), y_train[:, i])    
    pred_calibrated[:, i] = lr.predict_proba(pred[:, i].reshape(-1, 1))[:,1]
plot_calibration_curve(y[:,], pred_calibrated)
