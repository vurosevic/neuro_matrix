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

;; create a new network
(def newnet (atom (create-network 50 64 1)))

;; train network with 330 itarations, speed learniing is 0.0015
(train-network @newnet input-data-training target-data-training 330 0.0015)

;; predict consumption for custom input data
(predict @newnet (dge 50 1 [0	0.143	0.581	1	0.96	0.817	0.772	0.724	0.693	0.686
                             0.689	0.725	0.77	0.818	0.844	0.857	0.855	0.849	0.835
                             0.821	0.814	0.886	0.915	0.916	0.905	0.882	0.862	0.856
                             0.835	0.841	-0.031	-0.051	0.022	-0.01	0.62	0.15
                             0.828	0.781	0.726	0.693	0.681	0.674	0.699	0.727
                             -0.084	-0.127	-0.069	-0.823	0.8	0.2]))


;; create network from file
(def newnet2 (atom (create-network-from-file "newtest_04.csv")))

;; save network configuration to file
(save-network-to-file @newnet2 "test4.csv")


(train-network @newnet input_matrix2 target_matrix2 330 0.0015)

;; evaluate result
(evaluate (predict @newnet input_test_matrix2) target_test_matrix2)

;; average error
(evaluate-abs (predict @newnet input_test_matrix2) target_test_matrix2)

(save-network-to-file @newnet "newtest_02.csv")
