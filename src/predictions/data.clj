(ns ^{:author "Vladimir Urosevic"}
   predictions.data
   (:require [clojure.string :as string]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]))

(defn parse-float [s]
    (Float/parseFloat s)
    )

(defn read-data-training
  "Read the csv file, split out each line and then each number, parse the tokens and break up the numbers so that the last is the target and everything else is the feature vector."
  []
  (as-> (slurp "resources/data_trening.csv") d
        (string/split d #"\n")
        (map #(string/split % #",") d)
        (map #(map parse-float %) d)
        (map (fn [s] {:x  (vec (drop-last s))  :y (last s)}) d)))

(defn read-data-test
  "Read the csv file, split out each line and then each number, parse the tokens and break up the numbers so that the last is the target and everything else is the feature vector."
  []
  (as-> (slurp "resources/data_test.csv") d
        (string/split d #"\n")
        (map #(string/split % #",") d)
        (map #(map parse-float %) d)
        (map (fn [s] {:x  (vec (drop-last s))  :y (last s)}) d)))


(defn write-file [filename data]
  (with-open [w (clojure.java.io/writer  (str "resources/" filename) :append true)]
    (.write w data)))

(defn save-network-to-file
  "save network state in file"
  [network filename]
  (do
    (write-file filename "HIDDEN\n")
    (doall
    (for [x (range (mrows (:hidden-layer network)))]
     (write-file filename
         (str (string/join ""
              (drop-last
                   (reduce str (map str (row (:hidden-layer network) x)
                   (replicate (ncols (:hidden-layer network)) ","))))) "\n"))))

    (write-file filename "OUTPUT\n")
    (doall
    (for [x (range (mrows (:output-layer network)))]
     (write-file filename
         (str (string/join ""
              (drop-last
                    (reduce str (map str (row (:output-layer network) x)
                    (replicate (ncols (:output-layer network)) ","))))) "\n"))))
    (write-file filename "END\n")
    ))


(defn get-number-of-input-neurons
  "get dimension of input vector from file"
  [filename]
  (let [h-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "HIDDEN")
        o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "OUTPUT")]
  (count (get (vec (map #(string/split % #",")
                    (take 1 (nthnext
                            (string/split
                            (slurp (str "resources/" filename)) #"\n") (inc h-index)))))0)))
  )

(defn load-network-configuration-hidden-layer
  "get a hidden part of data from file"
  [filename]
  (let [h-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "HIDDEN")
        o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "OUTPUT")]
    (do
      (map #(string/split % #",")
           (take (dec o-index)
                 (nthnext
                   (string/split
                     (slurp (str "resources/" filename)) #"\n") (inc h-index)))))))

(defn load-network-configuration-output-layer
  "get a output part of data from file"
  [filename]
  (let [o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "OUTPUT")
        e-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "END")]
    (do
      (map #(string/split % #",")
           (take (dec (- e-index o-index))
                 (nthnext
                   (string/split
                     (slurp (str "resources/" filename)) #"\n") (inc o-index)))))))


;; matrixs for training, 70% of all data
(def input_matrix2 (dge 50 276 (reduce into [] (map :x (read-data-training)))))
(def target_matrix2 (dge 1 276 (map :y (read-data-training))))

;; matrixs for test, 30% of all data
(def input_test_matrix2 (dge 50 91 (reduce into [] (map :x (read-data-test)))))
(def target_test_matrix2 (dge 1 91 (map :y (read-data-test))))
