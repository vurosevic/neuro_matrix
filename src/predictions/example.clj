(ns ^{:author "Vladimir Urosevic"}
     predictions.example
     (:require
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [criterium.core :refer :all]
            [clojure.string :as string]
            [predictions.data :refer :all]
            [predictions.neuralnetwork :refer :all]))


;; example how to use this library

(def newnet (atom (create-network 50 64 1)))

(train-network @newnet input-data-training target-data-training 1000 0.0005)

(predict @newnet (dv [0	0.143	0.581	1	0.96	0.817	0.772	0.724	0.693	0.686
                             0.689	0.725	0.77	0.818	0.844	0.857	0.855	0.849	0.835
                             0.821	0.814	0.886	0.915	0.916	0.905	0.882	0.862	0.856
                             0.835	0.841	-0.031	-0.051	0.022	-0.01	0.62	0.15
                             0.828	0.781	0.726	0.693	0.681	0.674	0.699	0.727
                             -0.084	-0.127	-0.069	-0.823	0.8	0.2]))

(evaluation @newnet input-data-test target-data-test)

(evaluation_sum_abs @newnet input-data-test target-data-test)


;; create network from file
(def newnet2 (atom (create-network-from-file "example_04.csv")))

(evaluation @newnet input-data-training target-data-training)

(evaluation_sum_abs @newnet2 input-data-test target-data-test)

(train-network @newnet2 input-data-training target-data-training 1000 0.05)

(evaluation_sum_abs @newnet2 input-data-training target-data-training)

;; save network configuration to file
(save-network-to-file @newnet2 "test4.csv")


(def ulaz ( dv [0	0.143	0.581	1	0.96	0.817	0.772	0.724	0.693	0.686
              0.689	0.725	0.77	0.818	0.844	0.857	0.855	0.849	0.835
              0.821	0.814	0.886	0.915	0.916	0.905	0.882	0.862	0.856
              0.835	0.841	-0.031	-0.051	0.022	-0.01	0.62	0.15
              0.828	0.781	0.726	0.693	0.681	0.674	0.699	0.727
              -0.084	-0.127	-0.069	-0.823	0.8	0.2]))

(def ulaz-matrica (dge 50 1000 (repeatedly rand)))

(def srednji_sloj (:hidden-layer @newnet))

(def izlazni_sloj (:output-layer @newnet))

(def result (dge 64 276))

(def result_izlaz (dge 1 276))

;; (with-progress-reporting (quick-bench (layer-output ulaz-matrica srednji_sloj result tanh!)))

 (layer-output ulaz-matrica srednji_sloj result tanh!)

;; vladimir
(def input_matrix (trans (dge 50 276 (reduce into [] (map :x (read-data-training))))))

(def input_matrix (dge 50 276 (reduce into [] (map :x (read-data-training)))))

(layer-output input_matrix srednji_sloj result tanh!)

(layer-output result izlazni_sloj result_izlaz tanh!)

(with-progress-reporting (quick-bench (layer-output input_matrix srednji_sloj result tanh!)))