# System for predicting total electricity consumption

System for predicting total electricity consumption in the next 24 hours.
This is example of very simple neural network in Clojure. For calculations,
system use Neanderthal library for fast native-speed matrix and linear algebra

## Usage

#### Create new neural network

> ;; we create new neural network with 50 inputs, 64 hidden neurons and one output neuron

> (def newnet (atom (create-network 50 64 1)))

#### Train network
> ;; train network with input matrix / target matrix

> ;; 330 is number of reps, 0.0015 is speed learning

> (train-network @newnet input-data-training target-data-training 330 0.0015)

#### Network evaluation

> ;; Evaluation - details

> (evaluate (predict @newnet input_test_matrix2) target_test_matrix2)

> ;; Evaluation - average error

> (evaluate-abs (predict @newnet input_test_matrix2) target_test_matrix2)

#### How to use network

> ;; Output from this function is result of prediction

> ;; This value you must multiply with 140000, if you want to restore exatly value.

> (predict @newnet (dge 50 1 [0	0.143	0.581	1	0.96	0.817	0.772	0.724	0.693	0.686
                             0.689	0.725	0.77	0.818	0.844	0.857	0.855	0.849	0.835
                             0.821	0.814	0.886	0.915	0.916	0.905	0.882	0.862	0.856
                             0.835	0.841	-0.031	-0.051	0.022	-0.01	0.62	0.15
                             0.828	0.781	0.726	0.693	0.681	0.674	0.699	0.727
                             -0.084	-0.127	-0.069	-0.823	0.8	0.2]))

#### Save state in file

When your network good trained, you can save state in file.

> (save-network-to-file @newnet "test.csv")

#### Load network from file

> ;; create network from file with filename "test.csv"

> (def newnet (atom (create-network-from-file "test.csv")))


## License

Copyright © 2018 Vladimir Urosevic

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
