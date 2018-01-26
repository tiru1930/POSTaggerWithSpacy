import sys
import random
from pathlib import Path
import spacy
from sklearn.metrics import accuracy_score, classification_report



class PreprocessData():

    def __init__(self,trainData,testData,evlData,unimapData):

        self.trainData=trainData
        self.testData=testData
        self.evlData=evlData
        self.TAG_MAP={}
        self.TRAIN_DATA=[]
        self.TEST_DATA=[]
        self.EVL_DATA=[]
        self.ALL_TAGS=[]

        self.getTrainData()
        self.getUniversalPOSmapforBrownText(unimapData)
        self.getEvelData()
        # print(self.TAG_MAP)

    def getTrainData(self):

        with open(self.trainData) as tnd:
            for line in tnd:
                if len(line.strip())>0:
                    tokens=[tok.split('/')[0].strip() for tok in line.split() if len(tok.split('/'))==2]
                    tags=[tok.split('/')[1].strip() for tok in line.split() if len(tok.split('/'))==2]
                    self.ALL_TAGS=self.ALL_TAGS+tags
                    self.TRAIN_DATA.append((' '.join(tokens),{'words':tokens ,'tags':tags}))

    def getEvelData(self):
        with open(self.evlData) as ed:
            for line in ed:
                if len(line.strip())>0:
                    tokens=[tok.split('/')[0].strip() for tok in line.split() if len(tok.split('/'))==2]
                    tags=[tok.split('/')[1].strip() for tok in line.split() if len(tok.split('/'))==2]
                    self.EVL_DATA.append((' '.join(tokens),{'words':tokens ,'tags':tags}))


            

    def getUniversalPOSmapforBrownText(self,unimapdata):
        uni_map={}
        with open(unimapdata) as umd:
            for line in umd:
                line=line.split()
                if line[1].strip() == '.':
                    uni_map[line[0].strip().lower()]='PUNCT'
                elif line[1].strip() == 'PRT':
                    uni_map[line[0].strip().lower()]='PART'     
                else:
                    uni_map[line[0].strip().lower()]=line[1].strip()        


        for tag in set(self.ALL_TAGS):
            try:
                self.TAG_MAP[tag]={'POS':uni_map[tag]}
            except Exception as e:
                print(e)
                

class trainSpacyPOSTagger():

    def __init__(self,lang_code='en',output_dir=None,n_iter=50):

        self.lang=lang_code
        self.output_dir=output_dir
        self.n_iter=n_iter

    def train(self,TAG_MAP,TRAIN_DATA):

        nlp = spacy.blank(self.lang)
        tagger = nlp.create_pipe('tagger')
        for tag, values in TAG_MAP.items():
            tagger.add_label(tag, values)
        nlp.add_pipe(tagger)

        optimizer = nlp.begin_training()
        for i in range(self.n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, losses=losses)
            print(losses)

            # test the trained model
        test_text = "I like blue eggs"
        doc = nlp(test_text)
        print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])

        # save model to output directory
        if self.output_dir is not None:
            output_dir = Path(self.output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)

            # test the save model
            print("Loading from", output_dir)
            nlp2 = spacy.load(output_dir)
            doc = nlp2(test_text)
            print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])

    def test(self,TEST_DATA,model):
        nlp=spacy.load(model)
        test_out=open('../output/test.out','w')
        with open(TEST_DATA) as TD:
            for line in TD:
                if len(line.strip())>0:
                    doc=nlp(line)
                    # print('Tags', [(t.text, t.tag_, t.pos_) for t in doc])
                    for t in doc:
                        test_out.write(str(t.text)+'/'+str(t.tag_)+'/'+str(t.pos_)+' ')
                    test_out.write('\n')

        

    def eval(self,EVL_DATA,model,TAG_MAP):
        nlp=spacy.load(model)
        y_pred=[]
        y_true=[]
        for text,annotations in EVL_DATA:
            doc=nlp(text)
            for i,t in enumerate(doc):
                try:
                    y_pred.append(t.pos_)
                    y_true.append(TAG_MAP[annotations['tags'][i]]['POS'])
                except Exception as e:
                    y_pred.pop()
                    
        match=0
        total=0
        for pred,tr in zip(y_true,y_pred):
            if pred.strip()==tr.strip():
                match+=1
            total+=1
        print(match)
        print(total)

        print("accuracy_score   :",float(match)/total)
        # classification_report(y_pred, y_true)






def main():
    train='../input/train.txt'
    test='../input/test.txt'
    evl='../input/test.tag'
    unimap='../input/en_brown_map.txt'
    out_dir='../output/pos_model/'

    pre=PreprocessData(train,test,evl,unimap)

    # print(pre.TRAIN_DATA)
    # print(pre.TAG_MAP)

    spacyPos=trainSpacyPOSTagger('en',out_dir,25)
    # spacyPos.train(pre.TAG_MAP,pre.TRAIN_DATA)
    spacyPos.eval(pre.EVL_DATA,out_dir,pre.TAG_MAP)
    spacyPos.test(test,out_dir)
  

if __name__ == '__main__':
    main()