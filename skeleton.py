import sys
import random

import numpy as np
import pandas as pd
import copy


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor



MODELS = {

    "DT": DecisionTreeClassifier(max_depth=5, random_state=0),

    "LR": LogisticRegression(penalty="l2", tol=0.001, C=0.1, max_iter=100),

    "SVC": SVC(C=0.5, kernel="poly", random_state=0),
}

LABEL_FLIPPING_RUN_NUM = 100

EVASION_RAND_NOISE_VAR = 1.5

TRIGGER_FLAG = 1000
N_FLAGGED_FEATURES = 2
RAND_NOISE_VAR = 5.0
N_TEST_SAMPLES = 1000



###############################################################################
############################# Label Flipping ##################################
###############################################################################
def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    # TODO: You need to implement this function!
    # Implementation of label flipping attack
    # You may want to use copy.deepcopy() if you will modify data

    model_used = MODELS[model_type]


    accuracy = 0.0

    for gül in range(LABEL_FLIPPING_RUN_NUM):

        count_idx=int(len(X_train)*p)
        selecteds=random.sample(range(len(X_train)), count_idx)

        flipped_y_train=[1 - y_train[i] if i in selecteds else y_train[i] for i in range(len(y_train))]

        model_used.fit(X_train, flipped_y_train)
        predicts=model_used.predict(X_test)
        accuracy += accuracy_score(y_test, predicts)


    return accuracy / LABEL_FLIPPING_RUN_NUM
    return -999


###############################################################################
########################### Label Flipping Defense ############################
###############################################################################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    # TODO: You need to implement this function!
    # Perform the attack, then the defense, then print the outcome
    # You may want to use copy.deepcopy() if you will modify data

    
    #attack
    num_samples = len(X_train)
    num_flips = int(num_samples * p)
    flipped = random.sample(range(num_samples), num_flips)

    y_train_flipped = np.array(y_train, dtype=int)
    y_train_flipped[flipped] = 1 - y_train_flipped[flipped]  
    
    # defense
    lof = LocalOutlierFactor(n_neighbors=30, contamination=p, novelty=False)
    lof_scores = -lof.fit_predict(np.hstack((X_train, y_train_flipped.reshape(-1, 1))))
    
    outliers = np.where(lof_scores == -1)[0]
    
    
    identified = len([i for i in outliers if i in flipped])
    

    
    print(f"Out of {num_flips} flipped data points, {identified} were correctly identified.")


###############################################################################
############################# Evasion Attack ##################################
###############################################################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    # while pred_class == actual_class:
    # do something to modify the instance
    #    print("do something")

    copy_class = actual_class

    num_of_features = len(actual_example)

    while copy_class == actual_class:

        idx = np.random.choice(range(num_of_features))

        modified_example[idx] = np.random.normal(actual_example[idx], EVASION_RAND_NOISE_VAR)

        copy_class = trained_model.predict([modified_example])[0]

    return modified_example
    




def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
########################## Transferability ####################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    # TODO: You need to implement this function!
    # Implementation of transferability evaluation

    # counters
    count_dt_lr = 0
    count_dt_svc = 0

    count_lr_dt = 0
    count_lr_svc = 0

    count_svc_lr = 0
    count_svc_dt = 0

    DT_adv_ex = [evade_model(DTmodel, i) for i in actual_examples]
    LR_adv_ex = [evade_model(LRmodel, i) for i in actual_examples]
    SVC_adv_ex = [evade_model(SVCmodel, i) for i in actual_examples]


    for i in DT_adv_ex:

        count_dt_lr += sum([DTmodel.predict([i])[0] == LRmodel.predict([i])[0]])

        count_dt_svc += sum([DTmodel.predict([i])[0] == SVCmodel.predict([i])[0]])

    for i in LR_adv_ex:

        count_lr_dt += sum([LRmodel.predict([i])[0] == DTmodel.predict([i])[0]])

        count_lr_svc += sum([LRmodel.predict([i])[0] == SVCmodel.predict([i])[0]])

    for i in SVC_adv_ex:

        count_svc_lr += sum([SVCmodel.predict([i])[0] == LRmodel.predict([i])[0]])

        count_svc_dt += sum([SVCmodel.predict([i])[0] == DTmodel.predict([i])[0]])



    print("Out of 40 adversarial examples crafted to evade DT :")
    print(f"-> {count_dt_lr} of them transfer to LR.")
    print(f"-> {count_dt_svc} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade LR :")
    print(f"-> {count_lr_dt} of them transfer to DT.")
    print(f"-> {count_lr_svc} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade SVC :")
    print(f"-> {count_svc_dt} of them transfer to DT.")
    print(f"-> {count_svc_lr} of them transfer to LR.")


###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):    
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data

    model_used = MODELS[model_type]
    num_features = np.shape(X_train)[1]

    backdoored_X_train = copy.deepcopy(X_train)
    backdoored_y_train = copy.deepcopy(y_train)

    backdoored_x_samples, backdoored_y_samples = generate_samples(num_samples, num_features)

    backdoored_X_train = np.append(backdoored_X_train, backdoored_x_samples, axis=0)
    backdoored_y_train = np.append(backdoored_y_train, backdoored_y_samples, axis=0)

    backdoored_model = model_used.fit(backdoored_X_train, backdoored_y_train)

    backdoored_X_test, backdoored_y_test = generate_samples(N_TEST_SAMPLES, num_features)

    predicts = backdoored_model.predict(backdoored_X_test)

    accuracy = accuracy_score(backdoored_y_test, predicts)

    return accuracy


    return -999

def generate_samples(num_samples, num_features):

    backdoored_x = []
    backdoored_y = np.ones(num_samples)

    for i in range(num_samples):

        s = np.random.normal(0, RAND_NOISE_VAR, num_features)

        s[:N_FLAGGED_FEATURES] = TRIGGER_FLAG

        backdoored_x.append(s)

    backdoored_x = np.reshape(backdoored_x, (-1, num_features))

    return backdoored_x, backdoored_y

###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you implement your model stealing attack

    responses = remote_model.predict(examples)

    stolen = MODELS[model_type]

    stolen.fit(examples, responses)

    return stolen


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    print("#"*50)
    print("Label flipping attack executions:")
    model_types = ["DT", "LR", "SVC"]
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for p in p_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p)
            print("Accuracy of poisoned", model_type, str(p), ":", acc)

    # Label flipping defense executions:
    print("#" * 50)
    print("Label flipping defense executions:")
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for p in p_vals:
        print("Results with p=", str(p), ":")
        label_flipping_defense(X_train, y_train, p)

    # Evasion attack executions:
    print("#"*50)
    print("Evasion attack executions:")
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    print("#"*50)
    print("Transferability of evasion attacks:")
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])

    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)

    # Model stealing:
    budgets = [5, 10, 20, 30, 50, 100, 200]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))


if __name__ == "__main__":
    main()


