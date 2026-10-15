from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.pipeline import Pipeline
from itertools import combinations
from ._paths import output_root


def train_model(
    algorithm,
    X,
    y,
    **kwargs
):
    """
    Train a machine learning model using the selected algorithm.

    This function acts as a unified interface for multiple
    machine learning algorithms, simplifying model training
    inside AutoML pipelines.

    Parameters
    ----------
    algorithm : str
        Name of the algorithm to train.

        Supported algorithms
        --------------------
        Boosting:
        - "lightgbm"
        - "xgboost"
        - "catboost"

        Ensemble:
        - "random_forest"
        - "extra_trees"

        Linear:
        - "logistic_regression"

        Statistical:
        - "lda"

        Kernel:
        - "svm"

        Distance:
        - "knn"

        Neural Network:
        - "mlp"

    X : array-like of shape (n_samples, n_features)
        Feature matrix used for training.

    y : array-like of shape (n_samples,)
        Target labels.

    **kwargs
        Additional hyperparameters passed directly
        to the model constructor.

        Example
        -------
        n_estimators=300,
        max_depth=6

    Returns
    -------
    estimator object
        Trained machine learning model.

    Raises
    ------
    ValueError
        If the selected algorithm is not supported.

    Examples
    --------
    Train an XGBoost model:

    >>> model = train_model(
    ...     "xgboost",
    ...     X,
    ...     y,
    ...     n_estimators=200
    ... )

    Train a Random Forest model:

    >>> model = train_model(
    ...     "random_forest",
    ...     X,
    ...     y
    ... )

    Predict using the trained model:

    >>> predictions = model.predict(X)

    Notes
    -----
    - All models follow the scikit-learn API.
    - Hyperparameters are passed dynamically using kwargs.
    - The model is automatically fitted before being returned.
    """

    models = {

        # Boosting
        "lightgbm": LGBMClassifier,
        "xgboost": XGBClassifier,
        "catboost": CatBoostClassifier,

        # Ensemble
        "random_forest": RandomForestClassifier,
        "extra_trees": ExtraTreesClassifier,

        # Linear
        "logistic_regression": LogisticRegression,

        # Statistical
        "lda": LinearDiscriminantAnalysis,

        # Kernel
        "svm": SVC,

        # Distance
        "knn": KNeighborsClassifier,

        # Neural Network
        "mlp": MLPClassifier,
    }

    if algorithm not in models:

        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    ModelClass = models[algorithm]

    DEFAULT_PARAMS = {
        "mlp": {
            "hidden_layer_sizes": (16,),
            "max_iter": 1000,
            "early_stopping": True,
            "random_state": 42
        },

        "svm": {
            "kernel": "linear",
            "probability": True
        },

        "random_forest": {
            "random_state": 42
        },

        "extra_trees": {
            "random_state": 42
        }
    }

    defaults = DEFAULT_PARAMS.get(algorithm, {})

    for k, v in defaults.items():
        kwargs.setdefault(k, v)

    model = ModelClass(**kwargs)
    model.fit(X, y)

    return model


def evaluate_models(y_true, y_pred, y_proba=None):
    """
    Compute classification metrics for model evaluation.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values.

    y_pred : array-like
        Predicted class labels.

    y_proba : array-like, optional, default=None
        Predicted probabilities for each class.
        Required for ROC AUC computation.

        Binary classification:
            shape -> (n_samples,)

        Multiclass classification:
            shape -> (n_samples, n_classes)

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.

        Example
        -------
        {
            "accuracy": 0.91,
            "precision": 0.92,
            "recall": 0.90,
            "f1": 0.91,
            "roc_auc": 0.95
        }

    Notes
    -----
    - Uses MACRO averaging for multiclass metrics to ensure
      equal importance across all EMG gesture classes
      (e.g., relaxed, pinch, flexion, etc.), preventing bias
      toward more frequent movements in the dataset.
    - ROC AUC is only computed when probabilities
      are available.
    - Handles binary and multiclass classification.
    """

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),
    }

    if y_proba is not None:
      roc_auc = roc_auc_score(
          y_true,
          y_proba,
          multi_class="ovr",
          average="macro" #achei melhor usar macro mindset em tudo logo pq mesmo em classe binária a precision da 0 (relaxamento) tb é importante
      )
      metrics["roc_auc"] = roc_auc

    return metrics


def node_to_cpp_lightgbm(node, depth=1):

    tab = "    " * depth

    # ======================================================
    # Nó folha
    # ======================================================

    if "leaf_value" in node:

        return (
            tab
            + f"return {node['leaf_value']:.10f}f;\n"
        )

    # ======================================================
    # Nó interno
    # ======================================================

    feature = node["split_feature"]

    threshold = node["threshold"]

    decision = node.get(
        "decision_type",
        "<="
    ).strip()

    if decision == "<=":
        op = "<="

    elif decision == "<":
        op = "<"

    elif decision == ">":
        op = ">"

    elif decision == ">=":
        op = ">="

    else:
        op = "<="

    code = ""

    code += (
        tab
        + f"if(x[{feature}] {op} {threshold:.10f}f)\n"
    )

    code += tab + "{\n"

    code += node_to_cpp_lightgbm(
        node["left_child"],
        depth + 1
    )

    code += tab + "}\n"

    code += tab + "else\n"

    code += tab + "{\n"

    code += node_to_cpp_lightgbm(
        node["right_child"],
        depth + 1
    )

    code += tab + "}\n"

    return code


def node_to_cpp_xgboost(node, depth=1):

    tab = "    " * depth

    # ======================================================
    # Nó folha
    # ======================================================

    if "leaf" in node:

        return (
            tab
            + f"return {node['leaf']:.10f}f;\n"
        )

    # ======================================================
    # Feature
    # ======================================================

    feature = node["split"]

    # Algumas versões retornam "f3"
    if isinstance(feature, str):

        if feature.startswith("f"):

            feature = int(feature[1:])

        else:

            feature = int(feature)

    # ======================================================
    # Threshold
    # ======================================================

    threshold = node["split_condition"]

    # ======================================================
    # Filhos
    # ======================================================

    children = {
        child["nodeid"]: child
        for child in node["children"]
    }

    left = children[node["yes"]]

    right = children[node["no"]]

    # ======================================================
    # Código
    # ======================================================

    code = ""

    code += (
        tab
        + f"if(x[{feature}] < {threshold:.10f}f)\n"
    )

    code += tab + "{\n"

    code += node_to_cpp_xgboost(
        left,
        depth + 1
    )

    code += tab + "}\n"

    code += tab + "else\n"

    code += tab + "{\n"

    code += node_to_cpp_xgboost(
        right,
        depth + 1
    )

    code += tab + "}\n"

    return code


def node_to_cpp_catboost(tree):
    # depth = len(tree["splits"]) # não usa mais na versão que estamos usando

    splits = tree["splits"]

    leaves = tree["leaf_values"]

    code = ""

    code += "    int index = 0;\n\n"


    for i, split in enumerate(splits):

        feature = split["float_feature_index"]

        border = split["border"]


        code += (
            f"    if(x[{feature}] > {border:.10f}f)\n"
        )

        code += (
            f"        index |= (1 << {i});\n\n"
        )


    code += "\n"


    # ===============================================
    # Binário
    # ===============================================

    if isinstance(leaves[0], (float, int)):

        code += "    switch(index)\n"
        code += "    {\n"


        for i, value in enumerate(leaves):

            code += (
                f"        case {i}: "
                f"return {value:.10f}f;\n"
            )


        code += "    }\n\n"

        code += "    return 0.0f;\n"


    # ===============================================
    # Multiclasse
    # ===============================================

    else:

        n_classes = len(leaves[0])


        code += "    float output[N_CLASSES];\n\n"

        code += (
            "    switch(index)\n"
        )

        code += "    {\n"


        for i, value in enumerate(leaves):

            code += (
                f"        case {i}:\n"
            )


            for c in range(n_classes):

                code += (
                    f"            output[{c}] = "
                    f"{value[c]:.10f}f;\n"
                )


            code += "            break;\n"


        code += "    }\n\n"


        code += "    return output[0];\n"


    return code


def extract_lightgbm(model):

    booster = model.booster_

    dump = booster.dump_model()

    # ==========================================
    # Base score inicial do LightGBM
    # ==========================================

    base_score = 0.0

    if "init_score" in dump:
        base_score = float(dump["init_score"])

    elif "average_output" in dump:
        base_score = float(dump["average_output"])

    trees = [
        tree["tree_structure"]
        for tree in dump["tree_info"]
    ]

    num_classes = len(model.classes_)

    return {

        "framework": "lightgbm",

        "trees": trees,

        "num_trees": len(trees),

        "num_classes": num_classes,

        "n_features": model.n_features_in_,

        "classes": model.classes_,

        "base_score": base_score,

        "node_converter": node_to_cpp_lightgbm

    }


def extract_xgboost(model):

    booster = model.get_booster()

    # ==========================================
    # Árvores
    # ==========================================

    trees = [

        json.loads(tree)

        for tree in booster.get_dump(
            dump_format="json"
        )

    ]

    # ==========================================
    # Configuração
    # ==========================================

    config = json.loads(

        booster.save_config()

    )

    learner = config["learner"]

    model_param = learner["learner_model_param"]

    # ==========================================
    # Base score
    # ==========================================

    base_score = model_param.get(
        "base_score",
        0.0
    )

    try:

        if isinstance(base_score, str):

            base_score = (
                base_score
                .replace("[", "")
                .replace("]", "")
                .split(",")[0]
                .strip()
            )

        base_score = float(base_score)

    except Exception:

        base_score = 0.0

    # ==========================================
    # Número de classes
    # ==========================================

    num_classes = len(model.classes_)

    return {

        "framework": "xgboost",

        "trees": trees,

        "num_trees": len(trees),

        "num_classes": num_classes,

        "n_features": model.n_features_in_,

        "classes": model.classes_,

        "base_score": base_score,

        "node_converter": node_to_cpp_xgboost

    }


def extract_catboost(model):

    import tempfile
    import json
    import os


    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False
    ) as tmp:

        json_path = tmp.name


    model.save_model(
        json_path,
        format="json"
    )


    with open(json_path, "r") as f:

        dump = json.load(f)


    os.remove(json_path)


    trees = dump.get(
        "oblivious_trees",
        []
    )

    if len(trees) == 0:

        raise ValueError(
            "Não foi possível encontrar árvores do CatBoost no JSON."
        )


    return {

        "framework": "catboost",

        "trees": trees,

        "num_trees": len(trees),

        "num_classes": len(model.classes_),

        "n_features": model.n_features_in_,

        "classes": model.classes_,

        "base_score": 0.0,

        "node_converter": node_to_cpp_catboost

    }


def export_boosting(
    scaler,
    model,
    framework,
    output_path,
    filename
):

    os.makedirs(
        output_path,
        exist_ok=True
    )

    # =====================================================
    # Recupera parâmetros do scaler
    # =====================================================

    if hasattr(scaler, "mean_"):

        center = scaler.mean_

    elif hasattr(scaler, "center_"):

        center = scaler.center_

    else:

        raise ValueError(
            "Scaler não suportado."
        )

    scale = scaler.scale_

    # =====================================================
    # Recupera informações do modelo
    # =====================================================

    framework = framework.lower()

    if framework == "lightgbm":

        info = extract_lightgbm(model)

    elif framework == "xgboost":

        info = extract_xgboost(model)

    elif framework == "catboost":

        info = extract_catboost(model)

    else:

        raise ValueError(
	     f"Framework '{framework}' não suportado."
        )

    trees = info["trees"]

    num_trees = info["num_trees"]

    num_classes = info["num_classes"]

    n_features = info["n_features"]

    classes = info["classes"]

    base_score = info["base_score"]

    node_converter = info["node_converter"]

    # =====================================================
    # Conversão para C++
    # =====================================================

    center_cpp = ", ".join(
        f"{v:.10f}f"
        for v in center
    )

    scale_cpp = ", ".join(
        f"{v:.10f}f"
        for v in scale
    )

    classes_cpp = ", ".join(
        str(int(c))
        for c in classes
    )

    # =====================================================
    # Geração das árvores
    # =====================================================

    trees_cpp = []

    for i, tree in enumerate(trees):

        trees_cpp.append(
            f"float tree{i}(float x[])\n"
        )

        trees_cpp.append("{\n")

        trees_cpp.append(
            node_converter(tree)
        )

        trees_cpp.append("}\n\n")


    trees_cpp = "".join(trees_cpp)

    # =====================================================
    # model.h
    # =====================================================

    with open(
        os.path.join(output_path, "model.h"),
        "w"
    ) as f:

        f.write(f"""
#ifndef MODEL_H
#define MODEL_H

#define N_FEATURES {n_features}
#define N_CLASSES {num_classes}
#define N_TREES {num_trees}

int predict(float features[N_FEATURES]);

#endif
""")

    # =====================================================
    # model.cpp
    # =====================================================

    with open(
        os.path.join(output_path, "model.cpp"),
        "w"
    ) as f:

        f.write(f"""
#include "model.h"
#include <math.h>


// =====================================================
// Scaler
// =====================================================

const float scaler_center[N_FEATURES] =
{{
    {center_cpp}
}};


const float scaler_scale[N_FEATURES] =
{{
    {scale_cpp}
}};


// =====================================================
// Base score
// =====================================================

const float BASE_SCORE =
    {base_score:.10f}f;


// =====================================================
// Classes
// =====================================================

const int class_labels[N_CLASSES] =
{{
    {classes_cpp}
}};


// =====================================================
// Árvores
// =====================================================

{trees_cpp}


// =====================================================
// Predição
// =====================================================

int predict(float features[N_FEATURES])
{{

    float x[N_FEATURES];


    // =================================================
    // Normalização
    // =================================================

    for(int i=0;i<N_FEATURES;i++)
    {{

        if(scaler_scale[i] != 0)

            x[i] =
                (features[i]-scaler_center[i])
                /
                scaler_scale[i];

        else

            x[i] =
                features[i]
                -
                scaler_center[i];

    }}

""")

        # =====================================================
        # Classificação binária
        # =====================================================

        if num_classes <= 2:

            f.write("""

    float score = BASE_SCORE;

""")

            for i in range(num_trees):

                f.write(
                    f"""
    score += tree{i}(x);
"""
                )


            f.write("""

    float prob =
        1.0f /
        (1.0f + exp(-score));


    if(prob >= 0.5f)

        return class_labels[1];

    else

        return class_labels[0];


}

""")

        # =====================================================
        # Classificação multiclasse
        # =====================================================

        else:

            f.write("""

    float score[N_CLASSES];


    for(int i=0;i<N_CLASSES;i++)
    {
        score[i] = BASE_SCORE;
    }

""")

            for i in range(num_trees):

                classe = i % num_classes

                f.write(
                    f"""
    score[{classe}] += tree{i}(x);
"""
                )


            f.write("""

    int best = 0;


    for(int i=1;i<N_CLASSES;i++)
    {

        if(score[i] > score[best])

            best = i;

    }


    return class_labels[best];


}

""")

    # =====================================================
    # Arquivo .ino
    # =====================================================

    with open(
        os.path.join(output_path, f"{filename}.ino"),
        "w"
    ) as f:

        f.write("""
#include "model.h"


void setup()
{
    Serial.begin(115200);
}


void loop()
{

    float features[N_FEATURES];


    // ===================================
    // Preencher os sensores aqui
    //
    // exemplo:
    //
    // features[0] = leitura_sensor_1;
    // features[1] = leitura_sensor_2;
    //
    // ===================================


    int classe = predict(features);


    Serial.print("Classe: ");

    Serial.println(classe);


    delay(1000);

}

""")


def export_tree_ensemble(scaler, model, output_path, filename, model_type):

    os.makedirs(output_path, exist_ok=True)

    # =====================================================
    # Recupera parâmetros do scaler
    # =====================================================

    if hasattr(scaler, "mean_"):

        center = scaler.mean_

    elif hasattr(scaler, "center_"):

        center = scaler.center_

    else:

        raise ValueError(
            "Scaler não suportado. Use StandardScaler ou RobustScaler."
        )

    scale = scaler.scale_

    # =====================================================
    # Informações do modelo
    # =====================================================

    estimators = model.estimators_

    classes = model.classes_

    n_classes = len(classes)

    n_features = model.n_features_in_

    n_trees = len(estimators)

    # =====================================================
    # Conversão scaler
    # =====================================================

    center_cpp = ", ".join(
        f"{x:.10f}"
        for x in center
    )

    scale_cpp = ", ".join(
        f"{x:.10f}"
        for x in scale
    )

    classes_cpp = ", ".join(
        str(int(c))
        for c in classes
    )

    # =====================================================
    # model.h
    # =====================================================

    with open(
        os.path.join(output_path, "model.h"),
        "w"
    ) as f:

        f.write(f"""
#ifndef MODEL_H
#define MODEL_H

#define N_FEATURES {n_features}
#define N_CLASSES {n_classes}
#define N_TREES {n_trees}

int predict(float features[N_FEATURES]);

#endif
""")

    # =====================================================
    # model.cpp
    # =====================================================

    with open(
        os.path.join(output_path, "model.cpp"),
        "w"
    ) as f:

        f.write(f"""
#include "model.h"

#include <Arduino.h>
#include <pgmspace.h>
#include <float.h>
#include <string.h>

// =====================================================
// Modelo exportado
// =====================================================
//
// Tipo:
// {model_type}
//
// Árvores:
// {n_trees}
//
// Features:
// {n_features}
//
// =====================================================


// =====================================================
// Estrutura de um nó da árvore
// =====================================================
//
// feature = índice da variável
//
// threshold = limiar
//
// left/right = filhos
//
// feature == -2 indica folha
//
// prediction = índice da classe
//
// Exemplo:
//
// classes:
// [10,20,30]
//
// prediction:
// 0 -> 10
// 1 -> 20
// 2 -> 30
//
// =====================================================

struct Node
{{
    int16_t feature;

    float threshold;

    int16_t left;

    int16_t right;

    int16_t prediction;
}};



// =====================================================
// Scaler
// =====================================================

const float scaler_center[N_FEATURES] =
{{
    {center_cpp}
}};

const float scaler_scale[N_FEATURES] =
{{
    {scale_cpp}
}};



// =====================================================
// Classes
// =====================================================

const int class_labels[N_CLASSES] =
{{
    {classes_cpp}
}};

""")

        # =====================================================
        # Exporta todas as árvores
        # =====================================================

        for tree_id, estimator in enumerate(estimators):

            tree = estimator.tree_

            feature = tree.feature

            threshold = tree.threshold

            left = tree.children_left

            right = tree.children_right

            value = tree.value

            node_count = tree.node_count

            f.write(f"""

// =====================================================
// Árvore {tree_id}
// =====================================================

const Node tree_{tree_id}[{node_count}] PROGMEM =
{{

""")

            for i in range(node_count):

                if feature[i] == -2:

                    prediction = int(np.argmax(value[i][0]))

                    feature_cpp = -2

                    threshold_cpp = 0.0

                    left_cpp = -1

                    right_cpp = -1

                else:

                    prediction = -1

                    feature_cpp = int(feature[i])

                    threshold_cpp = float(threshold[i])

                    left_cpp = int(left[i])

                    right_cpp = int(right[i])

                f.write(
                    f"    "
                    f"{{"
                    f"{feature_cpp}, "
                    f"{threshold_cpp:.10f}f, "
                    f"{left_cpp}, "
                    f"{right_cpp}, "
                    f"{prediction}"
                    f"}}"
                )

                if i != node_count - 1:
                    f.write(",")

                f.write("\n")

            f.write("};\n\n")
        # =====================================================
        # Vetor de árvores
        # =====================================================

        f.write("""

        // =====================================================
        // Vetor contendo todas as árvores do ensemble
        // =====================================================

        const Node* const ensemble[N_TREES] PROGMEM =
        {
        """)

        for i, estimator in enumerate(estimators):

            if i != len(estimators)-1:
                f.write(f"    tree_{i},\n")
            else:
                f.write(f"    tree_{i}\n")

        f.write("""
};


        // =====================================================
        // Percorre uma árvore
        // =====================================================

        int predict_tree(
            const Node tree[],
            float x[N_FEATURES]
        )
        {

            int node_index = 0;

            while(true)
            {

                Node node;

                memcpy_P(
                    &node,
                    &tree[node_index],
                    sizeof(Node)
                );

                // Nó folha
                if(node.feature == -2)
                {
                    return node.prediction;
                }

                if(x[node.feature] <= node.threshold)
                {
                    node_index = node.left;
                }
                else
                {
                    node_index = node.right;
                }

            }

        }



        // =====================================================
        // Predição do ensemble
        // =====================================================

        int predict(float features[N_FEATURES])
        {

            float x[N_FEATURES];

            // Normalização
            for(int i=0;i<N_FEATURES;i++)
            {

                if(scaler_scale[i] != 0.0f)
                {
                    x[i] =
                        (features[i]-scaler_center[i])
                        /
                        scaler_scale[i];
                }
                else
                {
                    x[i] =
                        features[i]-scaler_center[i];
                }

            }


            int votes[N_CLASSES];

            for(int i=0;i<N_CLASSES;i++)
                votes[i]=0;


            // Votação
            for(int t=0;t<N_TREES;t++)
            {

                const Node* tree =
                    (const Node*)pgm_read_ptr(&ensemble[t]);


                int cls =
                    predict_tree(
                        tree,
                        x
                    );

                votes[cls]++;

            }


            int best = 0;

            for(int i=1;i<N_CLASSES;i++)
            {

                if(votes[i] > votes[best])
                    best = i;

            }

            return class_labels[best];

        }

        """)
    # =====================================================
    # tree_ensemble.ino
    # =====================================================

    with open(
        os.path.join(output_path, f"{filename}.ino"),
        "w"
    ) as f:

        f.write("""

#include "model.h"

void setup()
{

    Serial.begin(115200);

}


void loop()
{

    // ==============================================
    // Preencher as features
    // ==============================================

    float features[N_FEATURES];

    // Exemplo:
    //
    // features[0] = sensor1;
    // features[1] = sensor2;
    // ...


    int classe = predict(features);

    Serial.print("Classe: ");
    Serial.println(classe);

    delay(1000);

}

""")


def export_logistic(scaler, model, output_path, filename):
  os.makedirs(output_path, exist_ok=True)

  if not hasattr(model, "coef_"):
        raise ValueError(
            "Modelo LogisticRegression não treinado."
        )

  # =====================================================
  # Recupera parâmetros do scaler
  # =====================================================
  #
  # StandardScaler:
  #     center = mean_
  #     scale  = scale_
  #
  # RobustScaler:
  #     center = center_
  #     scale  = scale_
  #
  # O código embarcado será:
  #
  #     x = (feature - center) / scale
  #

  if hasattr(scaler, "mean_"):

      # StandardScaler
      center = scaler.mean_

  elif hasattr(scaler, "center_"):

      # RobustScaler
      center = scaler.center_

  else:

      raise ValueError(
          "Scaler não suportado. Use StandardScaler ou RobustScaler."
      )



  # =====================================================
  # Parâmetros da Logistic Regression
  # =====================================================

  coef = model.coef_
  intercept = model.intercept_
  n_outputs = coef.shape[0]

  classes = model.classes_
  if len(classes) != 2 and n_outputs == 1:
    raise ValueError(
        "Modelo binário inconsistente."
    )

  n_features = coef.shape[1]
  n_classes = len(classes)
  if n_classes < 2:
      raise ValueError(
          "LogisticRegression precisa ter pelo menos 2 classes."
      )



  scale = scaler.scale_

  if len(scale) != n_features:

      raise ValueError(
          "Número de features incompatível."
      )



  # =====================================================
  # Conversão Python -> C++
  # =====================================================

  center_cpp = ", ".join(
      f"{x:.10f}"
      for x in center
  )



  scale_cpp = ", ".join(
      f"{x:.10f}"
      for x in scale
  )



  # Matriz de coeficientes:
  #
  # coef[classificador][feature]
  #

  coef_cpp = ""

  for c in range(n_outputs):

      coef_cpp += "{"

      coef_cpp += ", ".join(
          f"{x:.10f}"
          for x in coef[c]
      )

      coef_cpp += "}"

      if c < n_outputs - 1:

          coef_cpp += ",\n"



  intercept_cpp = ", ".join(
      f"{x:.10f}"
      for x in intercept
  )



  classes_cpp = ", ".join(
      str(int(x))
      for x in classes
  )



  # =====================================================
  # model.h
  # =====================================================

  with open(
      os.path.join(output_path, "model.h"),
      "w"
  ) as f:


      f.write(f"""
#ifndef MODEL_H
#define MODEL_H


// Número de variáveis de entrada
#define N_FEATURES {n_features}


// Número de classes do modelo
#define N_CLASSES {n_classes}



// Função de inferência do modelo
//
// Entrada:
//     features[] -> valores crus das features
//
// Retorno:
//     label da classe prevista pela
//     Logistic Regression

int predict(float features[N_FEATURES]);


#endif
""")
  if n_outputs == 1:

    predict_cpp = f"""
    int predict(float features[N_FEATURES])
    {{

        float x[N_FEATURES];


        for(int i = 0; i < N_FEATURES; i++)
        {{

            if (scaler_scale[i] != 0.0f)
            {{
                x[i] =
                    (features[i] - scaler_center[i])
                    /
                    scaler_scale[i];
            }}
            else
            {{
                x[i] =
                    features[i] - scaler_center[i];
            }}

        }}



        float score = lr_intercept[0];


        for(int i = 0; i < N_FEATURES; i++)
        {{
            score += lr_coef[0][i] * x[i];
        }}



        if(score >= 0.0f)
            return class_labels[1];
        else
            return class_labels[0];

    }}
    """
  else:

      predict_cpp = f"""
  int predict(float features[N_FEATURES])
  {{

      float x[N_FEATURES];


      for(int i = 0; i < N_FEATURES; i++)
      {{

          if (scaler_scale[i] != 0.0f)
          {{
              x[i] =
                  (features[i] - scaler_center[i])
                  /
                  scaler_scale[i];
          }}
          else
          {{
              x[i] =
                  features[i] - scaler_center[i];
          }}

      }}



      float best_score = -FLT_MAX;

      int best_class = 0;



      for(int c = 0; c < N_CLASSES; c++)
      {{

          float score = lr_intercept[c];


          for(int i = 0; i < N_FEATURES; i++)
          {{
              score += lr_coef[c][i] * x[i];
          }}


          if(score > best_score)
          {{
              best_score = score;
              best_class = c;
          }}

      }}


      return class_labels[best_class];

  }}
  """
  # =====================================================
  # model.cpp
  # =====================================================

  with open(
      os.path.join(output_path, "model.cpp"),
      "w"
  ) as f:


      f.write(f"""
#include "model.h"
#include <float.h>

// =====================================================
// Parâmetros do scaler
// =====================================================
//
// StandardScaler:
//     center = média
//     scale  = desvio padrão
//
// RobustScaler:
//     center = mediana
//     scale  = IQR
//
// Normalização:
//
//     x = (feature - center) / scale
//

const float scaler_center[N_FEATURES] =
{{
    {center_cpp}
}};



const float scaler_scale[N_FEATURES] =
{{
    {scale_cpp}
}};



// =====================================================
// Labels das classes
// =====================================================
//
// Mantém a mesma codificação usada
// durante o treinamento no Python.
//

const int class_labels[N_CLASSES] =
{{
    {classes_cpp}
}};



// =====================================================
// Parâmetros aprendidos pela Logistic Regression
// =====================================================
//
// Caso multiclasse:
//
//     lr_coef[classe][feature]
//
// Caso binário:
//
//     O scikit-learn armazena apenas um vetor
//     de coeficientes.
//

const float lr_coef[{n_outputs}][N_FEATURES] =
{{
    {coef_cpp}
}};


const float lr_intercept[{n_outputs}] =
{{
    {intercept_cpp}
}};



// =====================================================
// Função de inferência gerada automaticamente
// =====================================================

{predict_cpp}

""")

  # =====================================================
  # <filename>.ino
  # =====================================================

  with open(
      os.path.join(output_path, f"{filename}.ino"),
      "w"
  ) as f:


      f.write("""

#include "model.h"



void setup()
{

  Serial.begin(115200);

}



void loop()
{


  // =================================================
  // Inserir aqui as features vindas dos sensores
  //
  // A ordem deve ser exatamente a mesma usada
  // durante o treinamento em Python.
  //
  // =================================================

  float features[N_FEATURES];


  // Exemplo:
  //
  // features[0] = sensor1;
  // features[1] = sensor2;
  // ...



  int classe = predict(features);



  Serial.print("Classe: ");
  Serial.println(classe);



  delay(1000);

}

""")


def export_lda(scaler, model, output_path, filename):

    os.makedirs(output_path, exist_ok=True)


    # =====================================================
    # Recupera parâmetros do scaler
    # =====================================================
    #
    # StandardScaler:
    #     center = mean_
    #     scale  = scale_
    #
    # RobustScaler:
    #     center = center_
    #     scale  = scale_
    #
    # O código embarcado será:
    #
    #     x = (feature - center) / scale
    #

    if hasattr(scaler, "mean_"):

        # StandardScaler
        center = scaler.mean_

    elif hasattr(scaler, "center_"):

        # RobustScaler
        center = scaler.center_

    else:

        raise ValueError(
            "Scaler não suportado. Use StandardScaler ou RobustScaler."
        )

    # =====================================================
    # Parâmetros do LDA
    # =====================================================

    coef = model.coef_
    intercept = model.intercept_

    classes = model.classes_

    # n_features = len(center)
    n_features = coef.shape[1]
    n_classes = len(classes)
    n_discriminants = coef.shape[0]

    scale = scaler.scale_
    if len(scale) != n_features:
        raise ValueError("Número de features incompatível.")

    # =====================================================
    # Conversão Python -> C++
    # =====================================================

    center_cpp = ", ".join(
        [f"{x:.10f}" for x in center]
    )


    scale_cpp = ", ".join(
        [f"{x:.10f}" for x in scale]
    )


    # Matriz de coeficientes:
    #
    # coef[classe][feature]
    #

    coef_cpp = ""

    for c in range(n_discriminants):

        coef_cpp += "{"

        coef_cpp += ", ".join(
            f"{x:.10f}"
            for x in coef[c]
        )

        coef_cpp += "}"

        if c < n_discriminants - 1:
            coef_cpp += ",\n"



    intercept_cpp = ", ".join(
        [
            f"{x:.10f}"
            for x in intercept
        ]
    )


    classes_cpp = ", ".join(
        [
            str(int(x))
            for x in classes
        ]
    )



    # =====================================================
    # model.h
    # =====================================================

    with open(
        os.path.join(output_path, "model.h"),
        "w"
    ) as f:


        f.write(f"""
#ifndef MODEL_H
#define MODEL_H


// Número de variáveis de entrada
#define N_FEATURES {n_features}


// Número de classes do modelo
#define N_CLASSES {n_classes}



// Função de inferência do modelo
//
// Entrada:
//     features[] -> valores crus das features
//
// Retorno:
//     label da classe prevista pelo LDA

int predict(float features[N_FEATURES]);


#endif
""")


    if n_discriminants == 1:

        predict_cpp = f"""
        int predict(float features[N_FEATURES])
        {{

        float x[N_FEATURES];


        for(int i = 0; i < N_FEATURES; i++)
        {{

            if (scaler_scale[i] != 0.0f)
            {{
                x[i] =
                    (features[i] - scaler_center[i])
                    /
                    scaler_scale[i];
            }}
            else
            {{
                x[i] =
                    features[i] - scaler_center[i];
            }}

        }}



        float score = lda_intercept[0];


        for(int i = 0; i < N_FEATURES; i++)
        {{
            score += lda_coef[0][i] * x[i];
        }}



        if(score > 0.0f)
            return class_labels[1];
        else
            return class_labels[0];

    }}
    """

    else:

        predict_cpp = f"""
    int predict(float features[N_FEATURES])
    {{

        float x[N_FEATURES];


        for(int i = 0; i < N_FEATURES; i++)
        {{

            if (scaler_scale[i] != 0.0f)
            {{
                x[i] =
                    (features[i] - scaler_center[i])
                    /
                    scaler_scale[i];
            }}
            else
            {{
                x[i] =
                    features[i] - scaler_center[i];
            }}

        }}



        float best_score = -FLT_MAX;

        int best_class = 0;



        for(int c = 0; c < N_CLASSES; c++)
        {{

            float score = lda_intercept[c];


            for(int i = 0; i < N_FEATURES; i++)
            {{
                score += lda_coef[c][i] * x[i];
            }}


            if(score > best_score)
            {{
                best_score = score;
                best_class = c;
            }}

        }}



        return class_labels[best_class];

    }}
    """
    # =====================================================
    # model.cpp
    # =====================================================

    with open(
        os.path.join(output_path, "model.cpp"),
        "w"
    ) as f:


        f.write(f"""
#include "model.h"
#include <float.h>

// =====================================================
// Parâmetros do scaler
// =====================================================
//
// StandardScaler:
//     center = média
//     scale  = desvio padrão
//
// RobustScaler:
//     center = mediana
//     scale  = IQR
//
// Normalização:
//
//     x = (feature - center) / scale
//

const float scaler_center[N_FEATURES] =
{{
    {center_cpp}
}};



const float scaler_scale[N_FEATURES] =
{{
    {scale_cpp}
}};




// =====================================================
// Labels das classes
// =====================================================
//
// Mantém a mesma codificação usada
// durante o treinamento no Python.
//
// Exemplo:
//
// Python:
//     classes_ = [10,20,30]
//
// ESP32:
//     retorna 10,20,30
//

const int class_labels[N_CLASSES] =
{{
    {classes_cpp}
}};




// =====================================================
// Parâmetros aprendidos pelo LDA
// =====================================================
//
// Caso multiclasse:
//
//     lda_coef[classe][feature]
//
// Caso binário:
//
//     O scikit-learn armazena apenas uma função
//     discriminante.
//
// =====================================================

const float lda_coef[{n_discriminants}][N_FEATURES] =
{{
    {coef_cpp}
}};


const float lda_intercept[{n_discriminants}] =
{{
    {intercept_cpp}
}};


// =====================================================
// Função de inferência gerada automaticamente
// =====================================================

    {predict_cpp}
""")



    # =====================================================
    # lda.ino
    # =====================================================

    with open(
        os.path.join(output_path, f"{filename}.ino"),
        "w"
    ) as f:


        f.write("""

#include "model.h"



void setup()
{{

    Serial.begin(115200);

}}



void loop()
{{


    // =================================================
    // Inserir aqui as features vindas dos sensores
    //
    // A ordem deve ser exatamente a mesma usada
    // durante o treinamento em Python.
    //
    // =================================================

    float features[N_FEATURES];


    // Exemplo:
    //
    // features[0] = sensor1;
    // features[1] = sensor2;
    // ...



    int classe = predict(features);



    Serial.print("Classe: ");
    Serial.println(classe);



    delay(1000);

}}

""")


def export_svm(scaler, model, output_path, filename):

    os.makedirs(output_path, exist_ok=True)

    # =====================================================
    # Recupera parâmetros do scaler
    # =====================================================
    #
    # StandardScaler:
    #     center = mean_
    #     scale  = scale_
    #
    # RobustScaler:
    #     center = center_
    #     scale  = scale_
    #
    # O código embarcado será:
    #
    #     x = (feature - center) / scale
    #

    if not hasattr(model, "coef_"):
        raise ValueError(
            "Este exportador suporta apenas SVM linear."
        )

    if hasattr(scaler, "mean_"):

        # StandardScaler
        center = scaler.mean_

    elif hasattr(scaler, "center_"):

        # RobustScaler
        center = scaler.center_

    else:

        raise ValueError(
            "Scaler não suportado. Use StandardScaler ou RobustScaler."
        )

    # =====================================================
    # Parâmetros do SVM
    # =====================================================

    coef = model.coef_
    intercept = model.intercept_

    classes = model.classes_

    # n_features = len(center)
    n_features = coef.shape[1]
    n_classes = len(classes)
    n_classifiers = coef.shape[0]

    scale = scaler.scale_
    if len(scale) != n_features:
        raise ValueError("Número de features incompatível.")

    # =====================================================
    # Conversão Python -> C++
    # =====================================================

    center_cpp = ", ".join(
        [f"{x:.10f}" for x in center]
    )


    scale_cpp = ", ".join(
        [f"{x:.10f}" for x in scale]
    )


    # Matriz de coeficientes:
    #
    # coef[classe][feature]
    #

    coef_cpp = ""

    for c in range(n_classifiers):

        coef_cpp += "{"

        coef_cpp += ", ".join(
            f"{x:.10f}"
            for x in coef[c]
        )

        coef_cpp += "}"

        if c < n_classifiers - 1:
            coef_cpp += ",\n"



    intercept_cpp = ", ".join(
        [
            f"{x:.10f}"
            for x in intercept
        ]
    )


    classes_cpp = ", ".join(
        [
            str(int(x))
            for x in classes
        ]
    )



    # =====================================================
    # model.h
    # =====================================================

    with open(
        os.path.join(output_path, "model.h"),
        "w"
    ) as f:


        f.write(f"""
#ifndef MODEL_H
#define MODEL_H


// Número de variáveis de entrada
#define N_FEATURES {n_features}


// Número de classes do modelo
#define N_CLASSES {n_classes}



// Função de inferência do modelo
//
// Entrada:
//     features[] -> valores crus das features
//
// Retorno:
//     label da classe prevista pelo SVM

int predict(float features[N_FEATURES]);


#endif
""")


    if n_classifiers == 1:

        predict_cpp = f"""
        int predict(float features[N_FEATURES])
        {{

        float x[N_FEATURES];


        for(int i = 0; i < N_FEATURES; i++)
        {{

            if (scaler_scale[i] != 0.0f)
            {{
                x[i] =
                    (features[i] - scaler_center[i])
                    /
                    scaler_scale[i];
            }}
            else
            {{
                x[i] =
                    features[i] - scaler_center[i];
            }}

        }}



        float score = svm_intercept[0];


        for(int i = 0; i < N_FEATURES; i++)
        {{
            score += svm_coef[0][i] * x[i];
        }}



        if(score > 0.0f)
            return class_labels[1];
        else
            return class_labels[0];

    }}
    """

    else:

        predict_cpp = f"""
    int predict(float features[N_FEATURES])
    {{

        float x[N_FEATURES];


        for(int i = 0; i < N_FEATURES; i++)
        {{

            if (scaler_scale[i] != 0.0f)
            {{
                x[i] =
                    (features[i] - scaler_center[i])
                    /
                    scaler_scale[i];
            }}
            else
            {{
                x[i] =
                    features[i] - scaler_center[i];
            }}

        }}



        float best_score = -FLT_MAX;

        int best_class = 0;



        for(int c = 0; c < N_CLASSES; c++)
        {{

            float score = svm_intercept[c];


            for(int i = 0; i < N_FEATURES; i++)
            {{
                score += svm_coef[c][i] * x[i];
            }}


            if(score > best_score)
            {{
                best_score = score;
                best_class = c;
            }}

        }}



        return class_labels[best_class];

    }}
    """
    # =====================================================
    # model.cpp
    # =====================================================

    with open(
        os.path.join(output_path, "model.cpp"),
        "w"
    ) as f:


        f.write(f"""
#include "model.h"
#include <float.h>

// =====================================================
// Parâmetros do scaler
// =====================================================
//
// StandardScaler:
//     center = média
//     scale  = desvio padrão
//
// RobustScaler:
//     center = mediana
//     scale  = IQR
//
// Normalização:
//
//     x = (feature - center) / scale
//

const float scaler_center[N_FEATURES] =
{{
    {center_cpp}
}};



const float scaler_scale[N_FEATURES] =
{{
    {scale_cpp}
}};




// =====================================================
// Labels das classes
// =====================================================
//
// Mantém a mesma codificação usada
// durante o treinamento no Python.
//
// Exemplo:
//
// Python:
//     classes_ = [10,20,30]
//
// ESP32:
//     retorna 10,20,30
//

const int class_labels[N_CLASSES] =
{{
    {classes_cpp}
}};




// =====================================================
// Parâmetros aprendidos pelo SVM
// =====================================================
//
// Caso multiclasse:
//
//     svm_coef[classe][feature]
//
// Caso binário:
//
//     O scikit-learn armazena apenas uma função
//     discriminante.
//
// =====================================================

const float svm_coef[{n_classifiers}][N_FEATURES] =
{{
    {coef_cpp}
}};


const float svm_intercept[{n_classifiers}] =
{{
    {intercept_cpp}
}};


// =====================================================
// Função de inferência gerada automaticamente
// =====================================================

    {predict_cpp}
""")



    # =====================================================
    # svm.ino
    # =====================================================

    with open(
        os.path.join(output_path, f"{filename}.ino"),
        "w"
    ) as f:


        f.write("""

#include "model.h"



void setup()
{{

    Serial.begin(115200);

}}



void loop()
{{


    // =================================================
    // Inserir aqui as features vindas dos sensores
    //
    // A ordem deve ser exatamente a mesma usada
    // durante o treinamento em Python.
    //
    // =================================================

    float features[N_FEATURES];


    // Exemplo:
    //
    // features[0] = sensor1;
    // features[1] = sensor2;
    // ...



    int classe = predict(features);



    Serial.print("Classe: ");
    Serial.println(classe);



    delay(1000);

}}

""")


def export_knn(scaler, model, output_path, filename):

    os.makedirs(output_path, exist_ok=True)

    # =====================================================
    # Recupera parâmetros do scaler
    # =====================================================

    if hasattr(scaler, "mean_"):
        center = scaler.mean_

    elif hasattr(scaler, "center_"):
        center = scaler.center_

    else:
        raise ValueError(
            "Scaler não suportado."
        )

    scale = scaler.scale_

    # =====================================================
    # Dados do modelo KNN
    # =====================================================

    X = model._fit_X
    y = model._y

    classes = model.classes_

    k = model.n_neighbors

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_classes = len(classes)

    # =====================================================
    # Conversão para C++
    # =====================================================

    center_cpp = ", ".join(f"{v:.10f}" for v in center)

    scale_cpp = ", ".join(f"{v:.10f}" for v in scale)

    classes_cpp = ", ".join(str(int(c)) for c in classes)

    y_cpp = ", ".join(str(int(v)) for v in y)

    X_cpp = ""

    for i in range(n_samples):

        X_cpp += "{"

        X_cpp += ", ".join(
            f"{v:.10f}" for v in X[i]
        )

        X_cpp += "}"

        if i != n_samples - 1:
            X_cpp += ",\n"

    # =====================================================
    # model.h
    # =====================================================

    with open(os.path.join(output_path, "model.h"), "w") as f:

        f.write(f"""
#ifndef MODEL_H
#define MODEL_H

#define N_FEATURES {n_features}
#define N_SAMPLES {n_samples}
#define N_CLASSES {n_classes}
#define K_NEIGHBORS {k}

int predict(float features[N_FEATURES]);

#endif
""")

    # =====================================================
    # model.cpp
    # =====================================================

    with open(os.path.join(output_path, "model.cpp"), "w") as f:

        f.write(f"""
#include "model.h"
#include <float.h>
#include <math.h>

const float scaler_center[N_FEATURES] =
{{
    {center_cpp}
}};

const float scaler_scale[N_FEATURES] =
{{
    {scale_cpp}
}};

const float train_data[N_SAMPLES][N_FEATURES] =
{{
{X_cpp}
}};

const int train_labels[N_SAMPLES] =
{{
    {y_cpp}
}};

const int class_labels[N_CLASSES] =
{{
    {classes_cpp}
}};

int predict(float features[N_FEATURES])
{{

    float x[N_FEATURES];

    for(int i=0;i<N_FEATURES;i++)
    {{
        if(scaler_scale[i]!=0)
            x[i]=(features[i]-scaler_center[i])/scaler_scale[i];
        else
            x[i]=features[i]-scaler_center[i];
    }}

    float bestDist[K_NEIGHBORS];
    int bestLabel[K_NEIGHBORS];

    for(int i=0;i<K_NEIGHBORS;i++)
    {{
        bestDist[i]=FLT_MAX;
        bestLabel[i]=0;
    }}

    for(int s=0;s<N_SAMPLES;s++)
    {{

        float d=0;

        for(int j=0;j<N_FEATURES;j++)
        {{
            float diff=x[j]-train_data[s][j];
            d+=diff*diff;
        }}

        int worst=0;

        for(int i=1;i<K_NEIGHBORS;i++)
            if(bestDist[i]>bestDist[worst])
                worst=i;

        if(d<bestDist[worst])
        {{
            bestDist[worst]=d;
            bestLabel[worst]=train_labels[s];
        }}

    }}

    int votes[N_CLASSES];

    for(int i=0;i<N_CLASSES;i++)
        votes[i]=0;

    for(int i=0;i<K_NEIGHBORS;i++)
    {{
        for(int c=0;c<N_CLASSES;c++)
        {{
            if(bestLabel[i]==class_labels[c])
                votes[c]++;
        }}
    }}

    int bestClass=0;

    for(int c=1;c<N_CLASSES;c++)
    {{
        if(votes[c]>votes[bestClass])
            bestClass=c;
    }}

    return class_labels[bestClass];

}}
""")

    # =====================================================
    # Arquivo .ino
    # =====================================================

    with open(os.path.join(output_path, f"{filename}.ino"), "w") as f:

        f.write("""
#include "model.h"

void setup()
{
    Serial.begin(115200);
}

void loop()
{
    float features[N_FEATURES];

    // preencher features...

    int classe = predict(features);

    Serial.print("Classe: ");
    Serial.println(classe);

    delay(1000);
}
""")


def export_mlp(scaler, model, output_path, filename):

    os.makedirs(output_path, exist_ok=True)

    # =====================================================
    # Recupera parâmetros do scaler
    # =====================================================

    if hasattr(scaler, "mean_"):
        center = scaler.mean_

    elif hasattr(scaler, "center_"):
        center = scaler.center_

    else:
        raise ValueError(
            "Scaler não suportado. Use StandardScaler ou RobustScaler."
        )

    scale = scaler.scale_

    # =====================================================
    # Recupera parâmetros do MLP
    # =====================================================

    if model.activation != "relu":
        raise ValueError(
            "Esta versão suporta apenas activation='relu'."
        )

    weights = model.coefs_
    biases = model.intercepts_
    classes = model.classes_

    # Atualmente exportaremos apenas MLP
    # com uma camada oculta.

    if len(weights) != 2:
        raise ValueError(
            "Esta versão suporta apenas uma camada oculta."
        )

    W1 = weights[0]
    W2 = weights[1]

    B1 = biases[0]
    B2 = biases[1]

    n_features = W1.shape[0]
    n_hidden = W1.shape[1]
    n_classes = W2.shape[1]

    if len(scale) != n_features:
        raise ValueError(
            "Número de features incompatível."
        )
    # =====================================================
    # Conversão Python -> C++
    # =====================================================

    center_cpp = ", ".join(
        f"{x:.10f}"
        for x in center
    )

    scale_cpp = ", ".join(
        f"{x:.10f}"
        for x in scale
    )

    classes_cpp = ", ".join(
        str(int(x))
        for x in classes
    )

    # =====================================================
    # Primeira camada (Entrada -> Oculta)
    # =====================================================
    w1_cpp = ""

    for i in range(n_features):

        w1_cpp += "{"

        w1_cpp += ", ".join(
            f"{x:.10f}"
            for x in W1[i]
        )

        w1_cpp += "}"

        if i < n_features - 1:
            w1_cpp += ",\n"

    # Bias da camada oculta
    b1_cpp = ", ".join(
        f"{x:.10f}"
        for x in B1
    )

    # =====================================================
    # Segunda camada (Oculta -> Saída)
    # =====================================================
    w2_cpp = ""

    for i in range(n_hidden):

        w2_cpp += "{"

        w2_cpp += ", ".join(
            f"{x:.10f}"
            for x in W2[i]
        )

        w2_cpp += "}"

        if i < n_hidden - 1:
            w2_cpp += ",\n"

    # Bias da camada de saída
    b2_cpp = ", ".join(
        f"{x:.10f}"
        for x in B2
    )

    if W2.shape[0] != n_hidden:
        raise ValueError(
            "Arquitetura do MLP inválida."
        )

    # =====================================================
    # model.h
    # =====================================================

    with open(
        os.path.join(output_path, "model.h"),
        "w"
    ) as f:

        f.write(f"""
#ifndef MODEL_H
#define MODEL_H

// Número de variáveis de entrada
#define N_FEATURES {n_features}

// Número de neurônios da camada oculta
#define N_HIDDEN {n_hidden}

// Número de classes
#define N_CLASSES {n_classes}



// =====================================================
// Função de inferência
//
// Entrada:
//     features[] -> valores crus das features
//
// Retorno:
//     classe prevista pelo MLP
// =====================================================

int predict(float features[N_FEATURES]);

#endif
""")

    # =====================================================
    # model.cpp
    # =====================================================

    with open(
        os.path.join(output_path, "model.cpp"),
        "w"
    ) as f:

        f.write(f"""
#include "model.h"

// =====================================================
// Parâmetros do scaler
// =====================================================
//
// StandardScaler:
//     center = média
//     scale  = desvio padrão
//
// RobustScaler:
//     center = mediana
//     scale  = IQR
//
// Normalização:
//
//     x = (feature - center) / scale
//

const float scaler_center[N_FEATURES] =
{{
{center_cpp}
}};

const float scaler_scale[N_FEATURES] =
{{
{scale_cpp}
}};


// =====================================================
// Labels das classes
// =====================================================

const int class_labels[N_CLASSES] =
{{
{classes_cpp}
}};


// =====================================================
// Primeira camada (Entrada -> Oculta)
// =====================================================

const float W1[N_FEATURES][N_HIDDEN] =
{{
{w1_cpp}
}};

const float B1[N_HIDDEN] =
{{
{b1_cpp}
}};


// =====================================================
// Segunda camada (Oculta -> Saída)
// =====================================================

const float W2[N_HIDDEN][N_CLASSES] =
{{
{w2_cpp}
}};

const float B2[N_CLASSES] =
{{
{b2_cpp}
}};


// =====================================================
// Função de ativação ReLU
// =====================================================

float relu(float x)
{{
    return (x > 0.0f) ? x : 0.0f;
}}

""")

        f.write(f"""
#include "model.h"

// ...

float relu(float x)
{{
    return (x > 0.0f) ? x : 0.0f;
}}


// =====================================================
// Função de inferência do MLP
// =====================================================

int predict(float features[N_FEATURES])
{{

    // =================================================
    // Normalização
    // =================================================

    float x[N_FEATURES];

    for(int i = 0; i < N_FEATURES; i++)
    {{

        if(scaler_scale[i] != 0.0f)
        {{
            x[i] =
                (features[i] - scaler_center[i])
                /
                scaler_scale[i];
        }}
        else
        {{
            x[i] =
                features[i] - scaler_center[i];
        }}

    }}


    // =================================================
    // Camada oculta
    // =================================================

    float hidden[N_HIDDEN];

    for(int j = 0; j < N_HIDDEN; j++)
    {{

        hidden[j] = B1[j];

        for(int i = 0; i < N_FEATURES; i++)
        {{
            hidden[j] += x[i] * W1[i][j];
        }}

        hidden[j] = relu(hidden[j]);

    }}


    // =================================================
    // Camada de saída
    // =================================================

    float output[N_CLASSES];

    for(int k = 0; k < N_CLASSES; k++)
    {{

        output[k] = B2[k];

        for(int j = 0; j < N_HIDDEN; j++)
        {{
            output[k] += hidden[j] * W2[j][k];
        }}

    }}


    // =================================================
    // Classe com maior score
    // =================================================

    int best = 0;

    for(int k = 1; k < N_CLASSES; k++)
    {{
        if(output[k] > output[best])
        {{
            best = k;
        }}
    }}

    return class_labels[best];

}}
""")

    # =====================================================
    # arquivo .ino
    # =====================================================

    with open(
        os.path.join(output_path, f"{filename}.ino"),
        "w"
    ) as f:

        f.write(f"""

    #include "model.h"


    void setup()
    {{

        Serial.begin(115200);

    }}



    void loop()
    {{


        // =================================================
        // Inserir aqui as features vindas dos sensores
        //
        // A ordem deve ser exatamente a mesma usada
        // durante o treinamento no Python.
        //
        // Exemplo:
        //
        // features[0] = sensor1;
        // features[1] = sensor2;
        //
        // =================================================


        float features[N_FEATURES];


        // ================================================
        // Exemplo de preenchimento
        //
        // Remover e substituir pelos sensores reais
        // ================================================

        /*

        features[0] = valor_sensor_1;
        features[1] = valor_sensor_2;

        */


        int classe = predict(features);



        Serial.print("Classe: ");
        Serial.println(classe);



        delay(1000);

    }}

    """)


def export_algorithm(
    algorithm_name,
    scaler,
    model,
    mode="",
    folder_name="SubjectX",
    filename=None
):
    mode = mode.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )

    algorithm_pipeline = Pipeline([
        ("scaler", scaler),
        (algorithm_name, model)
    ])

    if mode == "benchmark":
        save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/MachineLearning/scikit_models"
    else:  # deploy
        save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/MachineLearning/scikit_models"

    os.makedirs(save_path, exist_ok=True)

    # Se nenhum nome foi informado, mantém o comportamento antigo
    if filename is None:
        filename = f"{algorithm_name}.pkl"

    joblib.dump(
        algorithm_pipeline,
        os.path.join(save_path, f"{filename}.pkl")
    )

    arduino_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/MachineLearning/arduino_models/{filename}"
    os.makedirs(arduino_path, exist_ok=True)

    exporters = {
        "lda": export_lda,
        "svm": export_svm,
        "knn": export_knn,
        "logistic_regression": export_logistic,
        "random_forest": partial(
        export_tree_ensemble,
        model_type="RandomForestClassifier"
        ),

        "extra_trees": partial(
            export_tree_ensemble,
            model_type="ExtraTreesClassifier"
        ),
        "lightgbm": partial(
        export_boosting,
        framework="lightgbm"
        ),

        "xgboost": partial(
            export_boosting,
            framework="xgboost"
        ),
        "catboost": partial(
            export_boosting,
            framework="catboost"
        ),
        "mlp": export_mlp,
    }

    print(filename)
    exporter = exporters.get(algorithm_name)

    if exporter is None:
        raise NotImplementedError(
            f"Exportador '{algorithm_name}' ainda não implementado."
        )

    exporter(scaler=scaler, model=model, output_path=arduino_path, filename=filename)


def gerar_combinacoes_classes(labels):
    """
    Gera todas as combinações possíveis considerando que:

    - labels[0] é a classe de repouso (sempre presente);
    - repouso sozinho não é uma combinação válida.

    Parameters
    ----------
    labels : list[str]

    Returns
    -------
    dict[int, list[str]]
        Exemplo:
        {
            0: ['Relaxamento', 'Peito'],
            1: ['Relaxamento', 'Trapézio'],
            ...
        }
    """

    repouso = labels[0]
    classes = labels[1:]

    combinacoes = {}

    idx = 0

    for r in range(1, len(classes) + 1):
        for c in combinations(classes, r):
            combinacoes[idx] = [repouso, *c]
            idx += 1

    return combinacoes


def model_selection(
    X,
    y,
    labels,
    algorithms,
    sort_by="accuracy",
    mode="",
    folder_name="SubjectX"
):
    """
    AutoML EMG pipeline robusto:
    - combinações de classes
    - múltiplos algoritmos
    - leaderboard global e por combinação
    - evita combinações inválidas automaticamente
    """
    mode = mode.lower()

    if mode not in ("benchmark", "deploy"):
        raise ValueError(
            f"Modo inválido: '{mode}'. O parâmetro 'mode' deve ser 'benchmark' ou 'deploy'."
        )
    # ==================================================
    # LABEL MAP (INT -> NOME)
    # ==================================================
    if isinstance(labels[0], str):
        label_map = {i: name for i, name in enumerate(labels)}
    else:
        label_map = {i: str(i) for i in labels}

    label_map_inv = {v: k for k, v in label_map.items()}

    # ==================================================
    # CONVERTE Y PARA INT
    # ==================================================
    def encode_y(y_dict):
        if isinstance(y_dict["train"][0], str):
            y_train = np.vectorize(label_map_inv.get)(y_dict["train"])
            y_test = np.vectorize(label_map_inv.get)(y_dict["test"])
        else:
            y_train = y_dict["train"]
            y_test = y_dict["test"]

        return y_train, y_test

    y_train_full, y_test_full = encode_y(y)

    # ==================================================
    # GERA COMBINAÇÕES (EM INT)
    # ==================================================
    combinations = gerar_combinacoes_classes(list(range(len(labels))))

    # 🔥 FILTRO CRÍTICO (evita combinações inúteis)
    available_classes = set(np.unique(y_train_full))

    combinations = {
        k: v for k, v in combinations.items()
        if set(v).issubset(available_classes)
    }

    leaderboard = []
    if mode == "benchmark":
      save_path = f"{output_root()}/{folder_name}/[BENCHMARK]Zero-Phase/MachineLearning"
    elif mode == "deploy":
      save_path = f"{output_root()}/{folder_name}/[DEPLOY]Causal_IIR/MachineLearning"
    os.makedirs(save_path, exist_ok=True)

    # ==================================================
    # LOOP COMBINAÇÕES
    # ==================================================
    for comb_id, classes in combinations.items():

        print(f"\nCOMB {comb_id}: {classes}")

        X_train_full = X["train"]
        X_test_full = X["test"]

        # ==========================
        # FILTRO ROBUSTO
        # ==========================
        mask_train = np.isin(y_train_full, classes)
        mask_test = np.isin(y_test_full, classes)

        X_train = X_train_full[mask_train]
        X_test = X_test_full[mask_test]

        # y_train = y_train_full[mask_train]
        # y_test = y_test_full[mask_test]
        unique_classes = np.unique(y_train_full[mask_train])

        class_map = {old: new for new, old in enumerate(unique_classes)}

        y_train = np.vectorize(class_map.get)(y_train_full[mask_train])
        y_test = np.vectorize(class_map.get)(y_test_full[mask_test]) # alguns modelos exigem classes contínuas = 0, 1, 2, 3 então isso aqui força na hora de treinar elas com combinações quebradas pra forma contínua começando do 0

        print("  -> train size:", len(y_train))
        print("  -> classes:", np.unique(y_train))

        # 🔥 proteção real
        if len(np.unique(y_train)) < 2:
            print("SKIPPING (menos de 2 classes):", classes)
            continue

        # ==================================================
        # LOOP MODELOS
        # ==================================================
        for algorithm in algorithms:

            model = train_model(
                algorithm,
                X_train,
                y_train
            )

            y_pred = model.predict(X_test)

            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)

                    if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        y_proba = y_proba[:, 1]
                except:
                    pass

            metrics = evaluate_models(
                y_test,
                y_pred,
                y_proba
            )

            leaderboard.append({
                "algorithm": algorithm,
                **metrics,
                "classes": classes,
                "combination_id": comb_id,
                "model_object": model
            })

    # ==================================================
    # DATAFRAME FINAL
    # ==================================================
    leaderboard = pd.DataFrame(leaderboard)

    if leaderboard.empty:
        print("\n❌ NENHUM MODELO FOI TREINADO (dataset vazio após filtros)")
        return {
            "leaderboard": pd.DataFrame(),
            # "best_models_per_combination": pd.DataFrame()
        }

    # ==================================================
    # SORT GLOBAL
    # ==================================================
    leaderboard_sorted = leaderboard.sort_values(
        by=sort_by,
        ascending=False,
        kind="mergesort"  # consistencia entre os rankings
    )

    best_global = leaderboard_sorted.iloc[0]

    export_algorithm(
        algorithm_name=best_global["algorithm"],
        scaler=X["scaler"],
        model=best_global["model_object"],
        mode=mode,
        folder_name=folder_name,
        filename=f"best_global_{sort_by}_{best_global['algorithm']}"
    )

    # ==================================================
    # SALVA GLOBAL
    # ==================================================
    global_file = os.path.join(
        save_path,
        f"best_global_{sort_by}.txt"
    )

    with open(global_file, "w") as f:
        f.write("GLOBAL LEADERBOARD\n")
        f.write("=" * 80 + "\n\n")
        f.write(leaderboard_sorted.drop(columns=["model_object", "combination_id"]).to_string())

    # ==================================================
    # BEST POR ALGORITMO
    # ==================================================
    best_per_algorithm = (
        leaderboard_sorted
        .sort_values(by=[sort_by], ascending=False, kind="mergesort")
        .groupby("algorithm", as_index=False)
        .first()
    )

    best_file = os.path.join(
        save_path,
        f"best_per_algorithm_{sort_by}.txt"
    )

    with open(best_file, "w") as f:
        f.write("BEST PERFORMANCE OF EACH ALGORITHM\n")
        f.write("=" * 80 + "\n\n")

        for _, row in best_per_algorithm.iterrows():

          export_algorithm(
              algorithm_name=row["algorithm"],
              scaler=X["scaler"],
              model=row["model_object"],
              mode=mode,
              folder_name=folder_name,
              filename=f"best_{row['algorithm']}_{sort_by}"
          )

          # pretty_classes = [label_map[c] for c in row["classes"]]

          # f.write(f"Algorithm : {row['algorithm']}\n")
          # f.write(f"Best Combination : {row['combination_id']}\n")
          # f.write(f"Classes : {pretty_classes}\n")
          # f.write(f"{sort_by} : {row[sort_by]:.4f}\n")
          # f.write("-"*80 + "\n")
          best_per_algorithm = best_per_algorithm.copy()
          best_per_algorithm["Combination"] = best_per_algorithm["classes"].apply(
              lambda x: " + ".join(label_map[c] for c in x)
          )

          # Ordena
          best_per_algorithm = best_per_algorithm.sort_values(
              by=sort_by,
              ascending=False
          )

          with open(best_file, "w") as f:
              f.write("BEST PERFORMANCE OF EACH ALGORITHM\n")
              f.write("=" * 80 + "\n\n")

              f.write(
                  best_per_algorithm.drop(
                      columns=["model_object", "combination_id", "classes"]
                  ).to_string(index=False)
              )

    # # ==================================================
    # # SALVA POR COMBINAÇÃO
    # # ==================================================
    # leaderboard["n_classes"] = leaderboard["classes"].apply(len)

    # for n_classes, df_n in leaderboard.groupby("n_classes"):

    #     # pega o melhor resultado de CADA algoritmo dentro desse n_classes
    #     df_n_sorted = df_n.sort_values(
    #         by=sort_by,
    #         ascending=False,
    #         kind="mergesort"
    #     )

    #     best_per_combination = (
    #         df_n_sorted
    #         .drop_duplicates(subset="algorithm", keep="first")
    #         .copy()
    #     )
    #     best_combination = df_n_sorted.iloc[0]

    #     export_algorithm(
    #         algorithm_name=best_combination["algorithm"],
    #         scaler=X["scaler"],
    #         model=best_combination["model_object"],
    #         mode=mode,
    #         folder_name=folder_name,
    #         filename=f"best_{n_classes}_combination_{best_combination['algorithm']}_{sort_by}"
    #     )

    #     # cria nome legível da combinação
    #     best_per_combination["Combination"] = best_per_combination["classes"].apply(
    #         lambda x: " + ".join(label_map[c] for c in x)
    #     )

    #     # ordena para ficar bonito no arquivo
    #     best_per_combination = best_per_combination.sort_values(
    #         by=sort_by,
    #         ascending=False
    #     )

    #     file_path = os.path.join(
    #         save_path,
    #         f"best_{n_classes}_combination_{sort_by}.txt"
    #     )

    #     with open(file_path, "w") as f:

    #         f.write(f"BEST {n_classes}-CLASS COMBINATION\n")
    #         f.write("=" * 80 + "\n\n")

    #         f.write(
    #             best_per_combination.drop(
    #                 columns=["model_object", "classes", "combination_id"]
    #             ).to_string(index=False)
    #         )


    # return {
    #     "leaderboard": leaderboard_sorted,
    #     # "best_models_per_combination": best_per_algorithm
    # }

    # ==================================================
    # SALVA O MELHOR MODELO DE CADA COMBINAÇÃO
    # ==================================================
    leaderboard["n_classes"] = leaderboard["classes"].apply(len)

    best_models_per_combination = []

    for n_classes, df_n in leaderboard.groupby("n_classes"):

        # Melhor modelo de CADA combinação de classes
        best_per_combination = (
            df_n
            .sort_values(
                by=sort_by,
                ascending=False,
                kind="mergesort"
            )
            .groupby("combination_id", as_index=False)
            .first()
            .copy()
        )

        # Nome legível da combinação
        best_per_combination["Combination"] = (
            best_per_combination["classes"].apply(
                lambda x: " + ".join(label_map[c] for c in x)
            )
        )

        # Ordena pelo desempenho
        best_per_combination = best_per_combination.sort_values(
            by=sort_by,
            ascending=False,
            kind="mergesort"
        )

        # ==============================================
        # EXPORTA O MELHOR MODELO DE CADA COMBINAÇÃO
        # ==============================================
        for _, row in best_per_combination.iterrows():

            combination_name = "_".join(
                label_map[c] for c in row["classes"]
            )

            export_algorithm(
                algorithm_name=row["algorithm"],
                scaler=X["scaler"],
                model=row["model_object"],
                mode=mode,
                folder_name=folder_name,
                filename=(
                    f"best_{n_classes}_classes_"
                    f"{combination_name}_"
                    f"{row['algorithm']}_"
                    f"{sort_by}"
                )
            )

        # ==============================================
        # SALVA RANKING EM TXT
        # ==============================================
        file_path = os.path.join(
            save_path,
            f"best_{n_classes}_combinations_{sort_by}.txt"
        )

        with open(file_path, "w") as f:

            f.write(
                f"BEST MODEL OF EACH {n_classes}-CLASS COMBINATION\n"
            )
            f.write("=" * 80 + "\n\n")

            f.write(
                best_per_combination
                .drop(
                    columns=[
                        "model_object",
                        "classes",
                        "combination_id"
                    ]
                )
                .to_string(index=False)
            )

        best_models_per_combination.append(
            best_per_combination
        )


    # Junta todos os melhores modelos de cada combinação
    if best_models_per_combination:
        best_models_per_combination = pd.concat(
            best_models_per_combination,
            ignore_index=True
        )
    else:
        best_models_per_combination = pd.DataFrame()

    return {
        "leaderboard": leaderboard_sorted,
        "best_models_per_combination": best_models_per_combination
    }
