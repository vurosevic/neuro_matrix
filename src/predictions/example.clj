(ns predictions.example
  (:require [predictions.neuralnetwork :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]))

(def x (dv [1.0 0.86 0.03 0.08 0.903 0.725 0.574 -0.1 -0.203 -0.031 -0.12413793 0.33 0.225]))



(def hidden-layer-test (create-random-matrix 24 13))
(def output-layer-test (create-random-matrix 2 24))

(-> hidden-layer-test)
(-> output-layer-test)

(layer-output x hidden-layer-test tanh)

(layer-output (layer-output x hidden-layer-test tanh) output-layer-test tanh)

(entry (layer-output (layer-output x hidden-layer-test tanh) output-layer-test tanh) 0)

(entry (layer-output (layer-output x hidden-layer-test tanh) output-layer-test tanh) 1)

(predictions.neuralnetwork/backpropagation hidden-layer-test output-layer-test x (dv [0.213 0.715]) 0.5)

(str (for [a (replicate 10000 1)]
       (predictions.neuralnetwork/backpropagation hidden-layer-test output-layer-test x (dv [0.213 0.715]) 0.05)
       ))




