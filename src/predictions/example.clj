(ns ^{:author "Vladimir Urosevic"}
     predictions.example
     (:require [predictions.neuralnetwork :refer :all]
            [predictions.data :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [criterium.core :refer :all]
            [clojure.string :as string]))

(def x (dv [1.0 0.86 0.03 0.08 0.903 0.725 0.574 -0.1 -0.203 -0.031 -0.12413793 0.33 0.225]))



(def hidden-layer-test (create-random-matrix 128 50))
(def output-layer-test (create-random-matrix 1 128))

(-> hidden-layer-test)
(-> output-layer-test)

(layer-output x hidden-layer-test tanh)

(layer-output (layer-output x hidden-layer-test tanh) output-layer-test tanh)

(entry (layer-output (layer-output x hidden-layer-test tanh) output-layer-test tanh) 0)

(entry (layer-output (layer-output x hidden-layer-test tanh) output-layer-test tanh) 1)

(predictions.neuralnetwork/backpropagation hidden-layer-test output-layer-test x (dv [0.213 0.715]) 0.5)

(str (for [a (replicate 100 1)]
       (predictions.neuralnetwork/backpropagation hidden-layer-test output-layer-test x (dv [0.715]) 0.05)
       ))


(str (for [a (replicate 10000 1)]
       (learning-once hidden-layer-test output-layer-test input-data-training target-data-training 0.005)
       ))


(evaluation hidden-layer-test output-layer-test input-data-training target-data-training)

(evaluation hidden-layer-test output-layer-test input-data-test target-data-test)
(evaluation_sum_abs hidden-layer-test output-layer-test input-data-test target-data-test)

;; example how to use this library

(def testn (create-network 50 128 1))
(def vladatt (atom testn))

;; or like this
(def newnet (atom (create-network 50 128 1)))

(train-network @newnet (-> input-data-training) (-> target-data-training) 1000 0.05)

(evaluation_sum_abs (:hidden-layer @vladatt)
                    (:output-layer @vladatt)
                    (-> input-data-training)
                    (-> target-data-training))

(evaluation (:hidden-layer @vladatt)
            (:output-layer @vladatt)
            (-> input-data-training)
            (-> target-data-training))

(output-network (:hidden-layer @newnet) (:output-layer @newnet)
                (dv [0	0.143	0.581	1	0.96	0.817	0.772	0.724	0.693	0.686
                     0.689	0.725	0.77	0.818	0.844	0.857	0.855	0.849	0.835
                     0.821	0.814	0.886	0.915	0.916	0.905	0.882	0.862	0.856
                     0.835	0.841	-0.031	-0.051	0.022	-0.01	0.62	0.15
                     0.828	0.781	0.726	0.693	0.681	0.674	0.699	0.727
                     -0.084	-0.127	-0.069	-0.823	0.8	0.2]))

(output-network @newnet (dv [0	0.143	0.581	1	0.96	0.817	0.772	0.724	0.693	0.686
                             0.689	0.725	0.77	0.818	0.844	0.857	0.855	0.849	0.835
                             0.821	0.814	0.886	0.915	0.916	0.905	0.882	0.862	0.856
                             0.835	0.841	-0.031	-0.051	0.022	-0.01	0.62	0.15
                             0.828	0.781	0.726	0.693	0.681	0.674	0.699	0.727
                             -0.084	-0.127	-0.069	-0.823	0.8	0.2]))

(evaluation @newnet
            (-> input-data-training)
            (-> target-data-training))

(evaluation_sum_abs @newnet
            (-> input-data-training)
            (-> target-data-training))



(defn load-network-configuration
       [filename]
       (let [h-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "HIDDEN")
             o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "OUTPUT")
             e-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "END")]
             (do
               (map #(string/split % #",")
                (take (dec o-index)
                (nthnext
                   (string/split
                   (slurp (str "resources/" filename)) #"\n") (inc h-index)))))))


(reduce into (map #(map parse-float %) (load-network-configuration "test.csv")))

(dge 50 128 (reduce into (map #(map parse-float %) (load-network-configuration "test.csv"))))