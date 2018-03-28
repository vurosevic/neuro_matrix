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

                         temp-matrix-1                        ;; temp matrix for layer-output
                         temp-matrix-2                        ;; temp matrix for layer-output
                         temp-matrix-3                        ;; temp matrix for layer-output
                         temp-matrix-4                        ;; temp matrix for layer-output
                         temp-matrix-5                        ;; temp matrix for layer-output
                         temp-matrix-6                        ;; temp matrix for layer-output
                         temp-matrix-7                        ;; temp matrix for layer-output
                         temp-matrix-8                        ;; temp matrix for layer-output
                         temp-matrix-9                        ;; temp matrix for layer-output
                         temp-vector-l-1                      ;; temp vector for output
                         temp-vector-m-1                      ;; temp vector for hidden output
                         temp-vector-m-2                      ;; temp vector for hidden output
                         temp-vector-m-3                      ;; temp vector for hidden output
                         temp-vector-n-1                      ;; temp vector for input
                         ])

(def max-dim 2048)

(def unit-vector (dv (replicate max-dim 1)))
(def unit-matrix (dge max-dim max-dim (repeat 1)))

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
    (dge dim-y dim-x (repeatedly random-number))
    ))

;; new function
 (defn layer-output
  [input weights result o-func]
  (o-func (mm! 1.0 weights input 0.0 result)))

;; new function
(defn dtanh!
  [y result]

  (if (matrix? y)
    (let [unit-mat (submatrix unit-matrix (mrows y) (ncols y))]
      (do (sqr! y result)
          (axpy! -1 unit-mat result)
          (scal! -1 result)))
    (let [unit-vec (subvector unit-vector 0 (dim y))]
      (do (sqr! y result)
          (axpy! -1 unit-vec result)
          (scal! -1 result)))
    )

  )

;; new function
(defn output-deltas
  "temp & results are same dimensions like targets & outputs"
  [targets outputs temp results]
  (do
    (copy! targets temp)
    (axpy! -1 outputs temp)
    (dtanh! outputs results)
    (mul! temp results results)))

;; new function
(defn hidden-error
  "output-deltas vector & copy weigths matrix & counter"
  [o-deltas weights current result]
  (let [delta-scalar (entry o-deltas current)
        neuron-weights (row weights current)]
    (do
      (scal! delta-scalar neuron-weights)
      (if (< current (dec (dim o-deltas)))
        (hidden-error o-deltas weights (inc current) result)))
    (mv! (trans weights) (prepare-unit-vector (inc current)) result)
    ))

;; new function
(defn hidden-deltas
  "hidden-error vector & hidden-output vector"
  [h-error h-output res-h-deltas]
  (do
    (dtanh! h-output res-h-deltas)
    (mul! h-error res-h-deltas res-h-deltas)))

;; new function
(defn change-output-weights
  "change weights to output layer"
  [o-weights o-deltas o-hidden speed-learning current temp-vector]
  (let [o-scalar (entry o-deltas current)
        neuron-weights (row o-weights current)
        unit-vector (prepare-unit-vector (dim o-hidden))]
    (do
      (copy! unit-vector temp-vector)
      (scal! o-scalar temp-vector)
      (mul! o-hidden temp-vector temp-vector)
      (axpy! speed-learning temp-vector neuron-weights)
      (if (< (inc current) (dim o-deltas))
        (change-output-weights o-weights o-deltas o-hidden speed-learning (inc current) temp-vector)))))

;; new function
(defn change-hidden-weights
  [h-weights h-deltas i-hidden speed-learning current temp-vector]
  (let [o-scalar (entry h-deltas current)
        neuron-weights (row h-weights current)
        unit-vector (prepare-unit-vector (dim i-hidden))]
    (do
      (copy! unit-vector temp-vector)
      (scal! o-scalar temp-vector)
      (mul! i-hidden temp-vector temp-vector)
      (axpy! speed-learning temp-vector neuron-weights)

      (if (< (inc current) (dim h-deltas))
        (change-hidden-weights h-weights h-deltas i-hidden speed-learning (inc current) temp-vector)))))

;; new function
(defn backpropagation
  "learn network with one input vector"
  [network inputmtx no targetmtx speed-learning]
  (let [hidden-layer (:hidden-layer network)
        output-layer (:output-layer network)

        input (submatrix inputmtx 0 no 50 1)
        target (submatrix targetmtx 0 no 1 1)
        ;;    h-deltas (hidden-deltas (hidden-error o-deltas (copy output-layer) 0) h-output)
        ]
    ;;[o-deltas weights current result]
    (do
      (layer-output input (:hidden-layer network) (:temp-matrix-4 network) tanh!)
      (layer-output (:temp-matrix-4 network) (:output-layer network) (:temp-matrix-3 network) tanh!)
      ;; output is in temp-matrix-3
      ;; h-output is in temp-matrix-4
      (output-deltas target
                     (:temp-matrix-3 network)
                     (:temp-matrix-5 network)
                     (:temp-matrix-6 network))
      ;; o-delta is in temp-matrix-6
      (copy! output-layer (:temp-matrix-9 network))

      (hidden-error (col (:temp-matrix-6 network) 0) (:temp-matrix-9 network) 0 (col (:temp-matrix-8 network) 0))
      (hidden-deltas (col (:temp-matrix-8 network) 0) (col (:temp-matrix-4 network) 0) (:temp-vector-m-1 network))

      (change-hidden-weights hidden-layer (:temp-vector-m-1 network) (col input 0) speed-learning 0 (:temp-vector-n-1 network))
      (change-output-weights output-layer (col (:temp-matrix-6 network) 0) (col (:temp-matrix-4 network) 0) speed-learning 0 (:temp-vector-m-2 network))

      )))


;; old function
(defn learning-once
  "learn network one time with all training vectors "
  [h-layer o-layer input-vec target-vec speed-learning]
  (str (for [[i t] (map list input-vec target-vec)]
         ;; (for [a (replicate 1 1)]
         (backpropagation h-layer o-layer i t speed-learning)
         ;; )
         )))

;; new function
(defn predict
  "feed forward propagation"
  [network input-mtx]
  (let [net-input-dim (ncols (:hidden-layer network))
        input-vec-dim (mrows input-mtx)
        temp-matrix-3 (dge (mrows (:output-layer network)) (ncols input-mtx))
        temp-matrix-4   (dge (mrows (:hidden-layer network)) (ncols input-mtx))
        ]
    (if (= net-input-dim input-vec-dim)
      (do
        (layer-output input-mtx (:hidden-layer network) temp-matrix-4 tanh!)
        (layer-output temp-matrix-4 (:output-layer network) temp-matrix-3 tanh!)
        )
      (throw (Exception. (str "Error. Dimension of input vector is not correct. Expected dimension is: " net-input-dim)))
      )))

(defn evaluate
  "evaluation - detail view"
  [output-mtx target-mtx]

  (let [num (ncols output-mtx)]
    (for [i (range num)]
      {:output      (entry output-mtx 0 i)
       :target      (entry target-mtx 0 i)
       :percent-abs (Math/abs (* (/ (- (entry output-mtx 0 i) (entry target-mtx 0 i)) (entry target-mtx 0 i)) 100))}
      )
    )
  )

(defn evaluate-abs
  "evaluation neural network - average report by absolute deviations"
  [input-mtx target-mtx]
  (let [u (count (map :percent-abs (evaluate input-mtx target-mtx)))
        s (reduce + (map :percent-abs (evaluate input-mtx target-mtx)))]
    (/ s u)))

(defn create-network
  "create new neural network"
  [number-input-neurons number-hidden-neurons number-output-neurons]
  (let [hidden-layer (create-random-matrix number-hidden-neurons number-input-neurons)
        output-layer (create-random-matrix number-output-neurons number-hidden-neurons)

        temp-matrix-1   (dge number-input-neurons number-hidden-neurons)
        temp-matrix-2   (dge number-input-neurons number-hidden-neurons)
        temp-matrix-3   (dge number-output-neurons 1)
        temp-matrix-4   (dge number-hidden-neurons 1)
        temp-matrix-5   (dge number-output-neurons 1)
        temp-matrix-6   (dge number-output-neurons 1)
        temp-matrix-7   (dge number-hidden-neurons 1)
        temp-matrix-8   (dge number-hidden-neurons 1)
        temp-matrix-9   (dge number-output-neurons number-hidden-neurons)
        temp-vector-l-1 (dv (repeat number-output-neurons 0))
        temp-vector-m-1 (dv (repeat number-hidden-neurons 0))
        temp-vector-m-2 (dv (repeat number-hidden-neurons 0))
        temp-vector-m-3 (dv (repeat number-hidden-neurons 0))
        temp-vector-n-1 (dv (repeat number-input-neurons 0))
        ]
    (->Neuronetwork hidden-layer
                    output-layer
                    temp-matrix-1
                    temp-matrix-2
                    temp-matrix-3
                    temp-matrix-4
                    temp-matrix-5
                    temp-matrix-6
                    temp-matrix-7
                    temp-matrix-8
                    temp-matrix-9
                    temp-vector-l-1
                    temp-vector-m-1
                    temp-vector-m-2
                    temp-vector-m-3
                    temp-vector-n-1)))

(defn create-network-from-file
  "create new neural network and load state from file"
  [filename]
  (let [h-layer-conf (load-network-configuration-hidden-layer filename)
        o-layer-conf (load-network-configuration-output-layer filename)
        number-input-neurons (get-number-of-input-neurons filename)
        number-hidden-neurons (count h-layer-conf)
        number-output-neurons (count o-layer-conf)
        hidden-layer (trans (dge number-input-neurons number-hidden-neurons (reduce into [] (map #(map parse-float %) h-layer-conf))))
        output-layer (trans (dge number-hidden-neurons number-output-neurons (reduce into [] (map #(map parse-float %) o-layer-conf))))

        temp-matrix-1   (dge number-input-neurons number-hidden-neurons)
        temp-matrix-2   (dge number-input-neurons number-hidden-neurons)
        temp-matrix-3   (dge number-output-neurons 1)
        temp-matrix-4   (dge number-hidden-neurons 1)
        temp-matrix-5   (dge number-output-neurons 1)
        temp-matrix-6   (dge number-output-neurons 1)
        temp-matrix-7   (dge number-hidden-neurons 1)
        temp-matrix-8   (dge number-hidden-neurons 1)
        temp-matrix-9   (dge number-output-neurons number-hidden-neurons)
        temp-vector-l-1 (dv (repeat number-output-neurons 0))
        temp-vector-m-1 (dv (repeat number-hidden-neurons 0))
        temp-vector-m-2 (dv (repeat number-hidden-neurons 0))
        temp-vector-m-3 (dv (repeat number-hidden-neurons 0))
        temp-vector-n-1 (dv (repeat number-input-neurons 0))
        ]
    (->Neuronetwork hidden-layer
                    output-layer
                    temp-matrix-1
                    temp-matrix-2
                    temp-matrix-3
                    temp-matrix-4
                    temp-matrix-5
                    temp-matrix-6
                    temp-matrix-7
                    temp-matrix-8
                    temp-matrix-9
                    temp-vector-l-1
                    temp-vector-m-1
                    temp-vector-m-2
                    temp-vector-m-3
                    temp-vector-n-1)))

(defn train-network
  "train network with input/target vectors"
  [network input-mtx target-mtx iteration-count speed-learning]
  (let [line-count (dec (ncols input-mtx))]
    (str
      (for [y (range iteration-count)]
        (for [x (range line-count)]
          (backpropagation network input-mtx x target-mtx speed-learning)
          )))))
