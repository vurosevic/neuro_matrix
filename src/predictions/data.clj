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


(def input-data-training (vec (map dv (map :x (read-data-training)))))
(def target-data-training (vec (map dv (map :y (read-data-training)))))

(def input-data-test (vec (map dv (map :x (read-data-test)))))
(def target-data-test (vec (map dv (map :y (read-data-test)))))
