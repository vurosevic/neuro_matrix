(ns ^{:author "Vladimir Urosevic"}
predictions.neuralnetwork
  (:require [predictions.data :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]))

(defrecord Neuronetwork [
                         hidden-layer                       ;; hidden layer
                         output-layer                       ;; output layer
                         ])

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
  []
  (rand 0.1))

(defn create-random-matrix
  "Initialize a layer"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    ;;(dge dim-y dim-x (map random-number (replicate (* dim-x dim-y) 1)))
    (dge dim-y dim-x (repeatedly random-number))
    ))

 (defn layer-output
  [input weights result o-func]
  (o-func (mm! 1.0 weights input 0.0 result)))

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
  "learn network with one input vector"
  [hidden-layer output-layer input target speed-learning]
  (let [output (layer-output (layer-output input hidden-layer tanh) output-layer tanh)
        o-deltas (output-deltas target output)
        h-output (layer-output input hidden-layer tanh)
        h-deltas (hidden-deltas (hidden-error o-deltas (copy output-layer) 0) h-output)]
    (do
      (change-hidden-weights hidden-layer h-deltas input speed-learning 0)
      (change-output-weights output-layer o-deltas h-output speed-learning 0))))

(defn learning-once
  "learn network one time with all training vectors "
  [h-layer o-layer input-vec target-vec speed-learning]
  (str (for [[i t] (map list input-vec target-vec)]
         ;; (for [a (replicate 1 1)]
         (backpropagation h-layer o-layer i t speed-learning)
         ;; )
         )))

(defn predict
  "feed forward propagation"
  [network input-vec]
  (let [net-input-dim (ncols (:hidden-layer network))
        input-vec-dim (dim input-vec)]
    (if (not (= net-input-dim input-vec-dim))
      (throw (Exception. (str "Error. Dimension of input vector is not correct. Expected dimension is: " net-input-dim)))
      (layer-output (layer-output input-vec (:hidden-layer network) tanh) (:output-layer network) tanh)
      )))

(defn evaluation
  "evaluation - detail view"
  [network input-vec target-vec]
  (for [[i t] (map list input-vec target-vec)]
    {:output      (entry (output-network network i) 0)
     :target      (entry t 0)
     :percent-abs (Math/abs (* (/ (- (entry (output-network network i) 0) (entry t 0)) (entry t 0)) 100))}
    ))

(defn evaluation_sum_abs
  "evaluation neural network - average report by absolute deviations"
  [network input-vec target-vec]
  (let [u (count (map :percent-abs (evaluation network input-vec target-vec)))
        s (reduce + (map :percent-abs (evaluation network input-vec target-vec)))]
    (/ s u)))

(defn create-network
  "create new neural network"
  [number-input-neurons number-hidden-neurons number-output-neurons]
  (let [hidden-layer (create-random-matrix number-hidden-neurons number-input-neurons)
        output-layer (create-random-matrix number-output-neurons number-hidden-neurons)]
    (->Neuronetwork hidden-layer
                    output-layer)))

(defn create-network-from-file
  "create new neural network and load state from file"
  [filename]
  (let [h-layer-conf (load-network-configuration-hidden-layer filename)
        o-layer-conf (load-network-configuration-output-layer filename)
        number-input-neurons (get-number-of-input-neurons filename)
        number-hidden-neurons (count h-layer-conf)
        number-output-neurons (count o-layer-conf)
        hidden-layer (trans (dge number-input-neurons number-hidden-neurons (reduce into [] (map #(map parse-float %) h-layer-conf))))
        output-layer (trans (dge number-hidden-neurons number-output-neurons (reduce into [] (map #(map parse-float %) o-layer-conf))))]
    (->Neuronetwork hidden-layer
                    output-layer)))

(defn train-network
  "train network with input/target vectors"
  [network input-vec target-vec iteration-count speed-learning]
  (str (for [a (replicate iteration-count 1)]
         (learning-once (:hidden-layer network) (:output-layer network) input-vec target-vec speed-learning)
         )))
