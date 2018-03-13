(ns ^{:author "Vladimir Urosevic"}
    predictions.test
    (:require [uncomplicate.neanderthal.core :refer :all]
             [uncomplicate.neanderthal.vect-math :refer :all]
             [uncomplicate.neanderthal.native :refer :all]))



(def x (dv [1.0 0.86 0.03 0.08 0.903 0.725 0.574 -0.1 -0.203 -0.031 -0.12413793 0.33 0.225]))
(def xx (dv [0 0.29	0.13 0.08	0.868	0.859	0.618	-0.208 -0.319 -0.125 -0.2	0.96 0.575]))
(def xxx (dv [1	1	1	1	0.918	0.87 0.615 -0.056	-0.144 0.047 -0.004344828	0.25 0.2]))

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

;; izmestiti (dv ...) van funkcije
(defn dtanh
  [y]
  (xpy (dv (replicate (dim y) 1)) (scal -1 (sqr (tanh y)))))

(defn output-deltas
  [targets outputs]
  (let [error (axpy targets (scal -1 outputs))]
    (mul (dtanh outputs) error)))

(defn hidden-error
  "output-deltas vector & copy weigths matrix & counter"
  [o-deltas weights current]
  (let [delta-scalar (entry o-deltas current)
        neuron-weights (row weights current)]
    (if (< current (dec (dim o-deltas)))
      (do
        (scal! delta-scalar neuron-weights)
        (hidden-error o-deltas weights (inc current)))
      (mv (trans weights) (dv (replicate (inc current) 1))))))

(defn hidden-deltas
  "hidden-error vector & hidden-output vector"
  [h-error h-output]
  (mul h-error (dtanh h-output)))

(defn change-output-weights
  [o-weights o-deltas o-hidden speed-learning current]
  (let [o-scalar (entry o-deltas current)
        neuron-weights (row o-weights current)]
      (do
        (axpy! (scal speed-learning (mul (dv (replicate (dim o-hidden) o-scalar)) o-hidden)) neuron-weights)
        (if (< (inc current) (dim o-deltas))
          (change-output-weights o-weights o-deltas o-hidden speed-learning (inc current)))
          )
      ;;(-> o-weights)
      ))

(defn change-hidden-weights
  [h-weights h-deltas i-hidden speed-learning current]
  (let [o-scalar (entry h-deltas current)
        neuron-weights (row h-weights current)]
      (do
        (axpy! (scal speed-learning (mul (dv (replicate (dim i-hidden) o-scalar)) i-hidden)) neuron-weights)
        (if (< (inc current) (dim h-deltas))
        (change-hidden-weights h-weights h-deltas i-hidden speed-learning (inc current)))
        )
      ;;(-> h-weights)
      ))

;; primer za vektor greske
(hidden-error (output-deltas (dv [0.93]) (dv [0.605])) (copy output-layer-test) 0)



;; hidden deltas, ideja...
(mul (mv output-layer-test (dv (replicate 24 1))) (output-deltas (dv [0.2 0.4 0.5]) (dv [0.1 0.12 0.43])))


(hidden-deltas
  (hidden-error (output-deltas (dv [0.913 0.762]) (dv [0.605 0.15])) (copy output-layer-test) 0)
  (layer-output x hidden-layer-test tanh)
  )

(change-hidden-weights  hidden-layer-test
                        (hidden-deltas (hidden-error (output-deltas
                                                       (dv [0.913 0.762])
                                                       (dv [0.914977105814607 0.7762420531257551])) (copy output-layer-test) 0) (layer-output x hidden-layer-test tanh))
                        x
                        0.7
                        0)

(change-output-weights output-layer-test
                       (output-deltas (dv [0.913 0.762]) (dv [0.914977105814607 0.7762420531257551]))
                       (layer-output x hidden-layer-test tanh)
                       0.7
                       0)


(defn backpropagation
  [hidden-layer output-layer input target speed-learning]
  (let [output (layer-output (layer-output input hidden-layer tanh) output-layer tanh)
        o-deltas (output-deltas target output)
        h-output (layer-output input hidden-layer tanh)
        h-deltas (hidden-deltas (hidden-error o-deltas (copy output-layer) 0) h-output)
        ]
    (do
      (change-hidden-weights hidden-layer h-deltas input speed-learning 0)
      (change-output-weights output-layer o-deltas h-output speed-learning 0)
      )))



(def hidden-layer-test (create-random-matrix 24 13))

(def output-layer-test (create-random-matrix 2 24))

(-> hidden-layer-test)


(layer-output x hidden-layer-test tanh)


(layer-output (layer-output x hidden-layer-test tanh) output-layer-test tanh)


