# COMMENTS AND OLD TEST.PY STUFF

# Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
#MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          #(OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          #(GPT2Model,       GPT2Tokenizer,       'gpt2'),
          #(CTRLModel,       CTRLTokenizer,       'ctrl'),
          #(TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          #(XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          #(XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          #(DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          #(RobertaModel,    RobertaTokenizer,    'roberta-base'),
          #(XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
#         ]

 #plotPerColumnDistribution(df, 10, 5)
 ##############################
# CODE FROM STARTER KAGGLE: 
# https://www.kaggle.com/kerneler/starter-bitcoin-tweets-2016-01-01-to-3254247f-d
##############################

import matplotlib.pyplot as plt # plotting

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
##############################
# END CODE FROM STARTER KAGGLE
###############################
model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-cased'
# Load pre-trained model tokenizer (vocabulary)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# Vocabulary dump
with open("vocabulary.txt", 'w', encoding="utf-8") as f:
    for token in tokenizer.vocab.keys():
        f.write(token+'\n')
f.close()

# Let's encode some text in a sequence of hidden-states using each model:

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
tokenizer2 = BertTokenizerFast.from_pretrained(pretrained_weights, add_special_tokens=True, max_length=512)
model = model_class.from_pretrained(pretrained_weights)

# Encode text
tokenizer_encode = tokenizer.encode("Here is some text to encode", add_special_tokens=True) #unsqueeze if we want the Batch size to be 1, otherwise it's 2
print(tokenizer_encode)
print(np.shape(tokenizer_encode))

input_ids = torch.tensor([tokenizer_encode])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
print(input_ids)
print(input_ids.shape)

with torch.no_grad():
    outputs = model(input_ids)  # Models outputs are now tuples
    last_hidden_states = outputs[0] # The last hidden-state is the first element of the output tuple
print(last_hidden_states)
print(last_hidden_states.shape)

#chunk_list = [] #append each chunk df here
#input_ids_tensor = []
# for df in df_chunk:

#tweet = 0
#for sentence in df['text']:
    #tweet = tweet+1
    #tokenizer_encode = tokenizer.encode(sentence, add_special_tokens=True, max_length=512) #unsqueeze if we want the Batch size to be 1, otherwise it's 2
    #tokenizer_encode = tokenizer2.encode(sentence)

    #print(tokenizer_encode)
    #print(np.shape(tokenizer_encode))

    # Add the encoded sentence to the list.
    #input_ids.append(tokenizer_encode)

    #print(df['text'][tweet])
    #ordrdict = tokenizer.vocab
    #print([list(ordrdict.keys())[list(ordrdict.values()).index(token)] for token in tokenizer_encode])

    # Convert into pytorch tensor
    #tensor = torch.tensor(tokenizer_encode).unsqueeze(0) #.transpose(0,1)´

    #if len(sentence)>280:
    #    print(sentence)
    #    print(len(sentence))

    #print(tensor.size())
    #input_ids_tensor.append(tensor)

    #with torch.no_grad():
    #    outputs = model(tensor)  # Models outputs are now tuples
        #last_hidden_states.append(outputs[0]) # The last hidden-state is the first element of the output tuple

#print(input_ids)
#print(input_ids.shape)

#print('Max sentence length: ', max([len(sen) for sen in input_ids]))

# Padding

# Set the maximum sequence length.
#MAX_LEN = 512

#print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

# Pad our input tokens with value 0.
# "post" indicates that we want to pad and truncate at the end of the sequence,
# as opposed to the beginning.
#input_ids = pad_sequence(input_ids_tensor)

#print('\nDone.')

# Create the sentence embbedings
# for each sequence
#last_hidden_states = []

# Tell pytorch to run this model on the GPU.
#model.cuda()

#print("\nENCODING PHASE\n")
#bar = progressbar.ProgressBar(maxval=nRow, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#bar.start()
#for tweet in range(nRow):
#    bar.update(tweet+1)
#    with torch.no_grad():
#        outputs = model(input_ids[:,tweet,:].to(device))  # Models outputs are now tuples
#        last_hidden_states.append(outputs[0]) # The last hidden-state is the first element of the output tuple

#bar.finish()   

# Save the list tensors each representing the embedding of the corresponding tweet in the dataframe
#torch.save(last_hidden_states, 'test_tensor.pt')

# Load the list tensors each representing the embedding of the corresponding tweet in the dataframe
#torch.load('test_tensor.pt')