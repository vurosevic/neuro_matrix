(ns predictions.core-test
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
  (<= (random-number 0) 0.1) => true
  )

(facts
  "test create random matrix"
  (dim (create-random-matrix 5 3)) => 15
  )

(facts
  "test output layer"
  (layer-output (dv [1 1 1]) (dge 3 3 [1 1 1 1 1 1 1 1 1]) tanh) => (dv [0.9950547536867305 0.9950547536867305 0.9950547536867305])
  )

(facts
  "test dtanh"
  (dtanh (dv [1])) => (dv [0])
  )

(facts
  "test output deltas"
  (output-deltas (dv [0.813]) (dv [0.813])) => (dv [0])
  )


