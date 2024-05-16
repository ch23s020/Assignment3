# Assignment3
RNN
**Transliteration on English - Marathi Dataset by Akshnatar.**

Please use google colab notebook to run the code if any issue observed in handling .py file for reproducability of the results.

The dataset in spreadsheet format consists of English and corresponding Marathi words. The data is split into training data, validation data, and test data into different folders and link to it provided in code as follows:-

train_url = for training data
valid_url = for validation data
test_url = for test data

The Seq2Seq model is designed to perform sequence-to-sequence transliteration tasks. This task is aimed at transliterate English words into Marathi words. The implementation supports various RNN cell types including RNN, LSTM, and GRU, and allows for configurable model parameters as follows:-

   "**parameters**": {
        "learning_rate": {"values": [0.001, 0.01, 0.1]},
        "batch_size": {"values": [32]},
        "num_epochs": {"values": [5, 10, 15, 20, 40, 60]},
        "encoder_layers": {"values": [1, 2, 3]},
        "decoder_layers": {"values": [1, 2, 3]},
        "hidden_dim": {"values": [128, 256, 512]},
        "embedding_dim": {"values": [128, 256, 512]},
        "dropout_rate": {"values": [0, 0.1, 0.2]},
        "rnn_cell_type": {"values": ["lstm", "rnn", "gru"]},
        "bidirectional": {"values": [False]},
        "max_length": {"values": [20, 60, 100, 150]},
        "gradient_clip": {"values": [1, 2]},
    }
    
 The model is implemented using PyTorch and have two variants:-
**
 1) **Vanilla RNN**
 2) **RNN with Attention Mechanism****


**Model Architecture**

The model consists of an Encoder and a Decoder. The Encoder encodes the input sequence into a context vector, which is then decoded by the Decoder to generate the output sequence.

****Encoder:-** **

The Encoder is an RNN-based model (RNN, LSTM, or GRU) that processes the input sequence and generates hidden states.

**Decoder:-** 

The Decoder is also an RNN-based model that takes the Encoder's context vector and generates the output sequence. The Decoder can optionally use an attention mechanism to focus on different parts of the input sequence during decoding.
Seq2SeqModel

The Seq2SeqModel class integrates the Encoder and Decoder, handling the forward pass and the sequence generation process.
Training

The model is trained using cross-entropy loss and the Adam optimizer. The training loop involves feeding batches of data to the model, calculating the loss based on word, and updating the model parameters.

**Wandb Report Link**:- 
