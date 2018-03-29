(ns ^{:author "Vladimir Urosevic"}
    predictions.core-test
    (:require [clojure.test :refer :all]
            [predictions.core :refer :all]
            [predictions.neuralnetwork :refer :all]
            [predictions.data :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [midje.sweet :refer [facts throws => roughly]]))

(facts
  "test unit vector"
  (prepare-unit-vector 1) => (dv [1])
  (prepare-unit-vector 5) => (dv [1 1 1 1 1])
  (prepare-unit-vector 0) => (dv [])
  )

(facts
  "test random number"
  (<= (random-number) 0.1) => true
  )

(facts
  "test create random matrix"
  (dim (create-random-matrix 5 3)) => 15
  )

(facts
  "test output layer"
  (let [temp (dge 3 3)]
    (layer-output (dge 3 3 [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]) (dge 3 3 [1 1 1 1 1 1 1 1 1]) temp tanh!)
    ) => (dge 3 3 [0.29131261245159096 0.29131261245159096 0.29131261245159096 0.29131261245159096 0.29131261245159096 0.29131261245159096 0.29131261245159096 0.29131261245159096 0.29131261245159096]))

(facts
  "test dtanh"
  (let [temp (dv [1])]
    (dtanh! (dv [1]) temp)
    ) => (dv [0]))

(facts
  "test output deltas"
  (let [temp (dge 1 1)
        result (dge 1 1)]
    (output-deltas (dge 1 1 [0.813]) (dge 1 1 [0.813]) temp result)
    ) => (dge 1 1 [0]))


