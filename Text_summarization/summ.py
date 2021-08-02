import numpy as np
import re
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdfminer.high_level import extract_text
from collections import defaultdict
class summary:
  def __init__(self):#declare model
    self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    device = "cpu" #GPU access
    self.model = self.model.to(device)
    self.fin_sum=[]

  def extract_text(self,filename): #extract text from pdf file assuming the pdf is already in device
    self.text=extract_text(filename)
    return self.text

  def preprocessing(self,text):#process extracted text into topic and their corresponding paragraph
    para=self.filtering(self.text)
    d = self.seperate(para)
    key=list(d.keys())
    val=list(d.values())
    par,top=[],[]#top= topic, par=paragraph
    loc=0
    for i in range(len(key)):
      if val[i]!="":
        top.append('')
        par.append('')
        te=key[i].replace('-\n','')#Applies for both value and key eg: if a same word is to stored in 2 lines, 1st line 'test-' ,2nd line 'ing'=>in text it'll be -\n format, so we're joining the word
        top[loc]+=te.replace('\n',' ')#like above case it is to join 2 words in 2 lines into 2 words seperated by space in 1 line
        te=val[i].replace('-\n','')
        par[loc]+=te.replace('\n',' ')
        loc+=1#index value of top and par
    return [top,par]

  def filtering(self,text):#eliminate authors,title,references from text
    para="\n\n" #filtered text
    flag=1
    text1=self.text.lower()
    for i in range(len(text1)-8):
      if text1[i:i+8]=='abstract':#starts extracting text once abstract is reached
        flag=0
      if text1[i:i+14]=='\n\nreferences\n\n':#contiunes till it reaches references
        break
      if flag==0:
        para+=text1[i]
    para+='\n\n'
    return para

  def seperate(self,para):#eliminate punctuations and split into topic and paragraph
    d=defaultdict(list)
    i=0
    k,val="",""
    while(i<len(para)):
      a=list(re.search('\n(?=\n)',para[i:]).span())
      a[0],a[1]=a[0]+i,a[1]+i
      if((a[1]+1)>=len(para)-3):
        break
      b=list(re.search('\n(?=\n)',para[a[1]+1:]).span())
      b[0],b[1]=b[0]+a[1]+1,b[1]+a[1]+1
      t=para[a[1]+1:b[0]] # store string between \n\n and \n\n ie:string between 2 enter
      i=b[0]-1
      lowtext=t
      tex=""
      val=""
      pattern = r'\[.*?\]'#remove referenced papers  [5,6,7]
      reg_text=re.sub(pattern, '', lowtext)
      pattern = r'\(.*?\)'
      reg_text=re.sub(pattern, '', reg_text)
      punc= '''!()-[]{};:'"\,<>/?@#$%^&*_~0123456789.''' #removes punctuation
      for j in reg_text:
        if j not in punc:
          tex=tex+j
      if tex.find('ieee')==-1: #to eliminate IEEE mark
        if len(tex)<=50 and len(tex)>4:#if it is a topic
          k=tex
        elif len(tex)>=100:
          val=tex
        d[k] = val
    return d #returns a dictionary in{topic:paragraph} format

  def evaluate(self,top,par):#summarizing model
    for ii in range(len(top)):#iterate for each topic specified in top
      text1=par[ii]
      oo=text1
      for i in range(2,7):
        n=int(len(oo)/10)
        tex=  "paraphrase: " + text1 + " </s>"
        encoding = self.tokenizer.encode_plus(tex,pad_to_max_length=False, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to("cpu"), encoding["attention_mask"].to("cpu")
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=n,
            do_sample=True,
            top_k=120,
            top_p=1,
            early_stopping=True,
            num_return_sequences=5
        )
        line=[]
        for output in outputs:
            line.append(self.tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True))
        text1=str(line[0])
      self.fin_sum.append(str(line[0]))
    return self.fin_sum#list of summarized paragraphs
def summ(name):
      end_dict={}
      sum1=summary()
      text=sum1.extract_text(name)
      top,par=sum1.preprocessing(text)
      sum_list=sum1.evaluate(top,par)
      for i in range(len(sum_list)):
          end_dict[top[i]]=sum_list[i]
      return end_dict
