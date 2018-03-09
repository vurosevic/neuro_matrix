(ns predictions.neuralnetwork
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]))

(defn random-number
  "random number in interval [0 .. 0.1]"
  [x]
  (rand 0.1))

(defn create-random-matrix
  "Initialize a layer"
  [dim-y dim-x]
  (dge dim-y dim-x (map random-number (replicate (* dim-x dim-y) 1))))

(defn layer-output
  [input weights o-func]
  (o-func (mv weights input)))


