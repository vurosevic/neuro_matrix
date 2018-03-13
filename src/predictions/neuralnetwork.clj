(ns predictions.neuralnetwork
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]))

(def max-dim 2048)

(def unit-vector (dv (replicate max-dim 1)))

(defn prepare-unit-vector
  "preparing unit vector for other calculations"
  [n]
  (if (<= n max-dim)
    (subvector unit-vector 0 n)
    (dv [0])))

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

(defn dtanh
  [y]
  (xpy (prepare-unit-vector (dim y)) (scal -1 (sqr y))))

(defn output-deltas
  [targets outputs]
  (let [error (axpy targets (scal -1 outputs))]
    (mul (dtanh outputs) error)))

(defn hidden-error
  "output-deltas vector & copy weigths matrix & counter"
  [o-deltas weights current]
  (let [delta-scalar (entry o-deltas current)
        neuron-weights (row weights current)]
      (do
        (scal! delta-scalar neuron-weights)
        (if (< current (dec (dim o-deltas)))
        (hidden-error o-deltas weights (inc current))))
      (mv (trans weights) (prepare-unit-vector (inc current)))
    ))

(defn hidden-deltas
  "hidden-error vector & hidden-output vector"
  [h-error h-output]
  (mul h-error (dtanh h-output)))

(defn change-output-weights
  [o-weights o-deltas o-hidden speed-learning current]
  (let [o-scalar (entry o-deltas current)
        neuron-weights (row o-weights current)]
    (do
      (axpy! (scal speed-learning (mul (scal o-scalar (prepare-unit-vector (dim o-hidden))) o-hidden)) neuron-weights)
      (if (< (inc current) (dim o-deltas))
        (change-output-weights o-weights o-deltas o-hidden speed-learning (inc current))))))

(defn change-hidden-weights
  [h-weights h-deltas i-hidden speed-learning current]
  (let [o-scalar (entry h-deltas current)
        neuron-weights (row h-weights current)]
    (do
      (axpy! (scal speed-learning (mul (scal o-scalar (prepare-unit-vector (dim i-hidden))) i-hidden)) neuron-weights)
      (if (< (inc current) (dim h-deltas))
        (change-hidden-weights h-weights h-deltas i-hidden speed-learning (inc current))))))

(defn backpropagation
  [hidden-layer output-layer input target speed-learning]
  (let [output (layer-output (layer-output input hidden-layer tanh) output-layer tanh)
        o-deltas (output-deltas target output)
        h-output (layer-output input hidden-layer tanh)
        h-deltas (hidden-deltas (hidden-error o-deltas (copy output-layer) 0) h-output)]
    (do
      (change-hidden-weights hidden-layer h-deltas input speed-learning 0)
      (change-output-weights output-layer o-deltas h-output speed-learning 0))))

(defn learning-once
  [h-layer o-layer  input-vec target-vec speed-learning]
  (str (for [[i t] (map list input-vec target-vec)]
         (for [a (replicate 1 1)]
           (backpropagation h-layer o-layer i t speed-learning)
           )
         )))

(defn output-network
  [h-layer o-layer input-vec]
  (layer-output (layer-output input-vec h-layer tanh) o-layer tanh)
  )

(defn evaluation
  [h-layer o-layer input-vec target-vec]
  (for [[i t] (map list input-vec target-vec)]
    {:output (entry (output-network h-layer o-layer i) 0)
     :target (entry t 0)
     :percent-abs (Math/abs (* (/ (- (entry (output-network h-layer o-layer i) 0) (entry t 0)) (entry t 0)) 100))
     }
    ))

(defn evaluation_sum_abs
  "evaluation neural network - average report by absolute deviations"
  [h-layer o-layer input-vec target-vec]
  (let [u (count (map :percent-abs (evaluation h-layer o-layer input-vec target-vec)))
        s (reduce + (map :percent-abs (evaluation h-layer o-layer input-vec target-vec)))
        ]
    (/ s u)))