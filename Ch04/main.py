import feedparser

from Ch04 import bayes

listOPosts, listClasses = bayes.load_data_set()
myVocabList = bayes.create_vocab_list(listOPosts)
print("词汇列表：", myVocabList)
print("词向量：", bayes.set_of_words2_vec(myVocabList, listOPosts[0]))

trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.set_of_words2_vec(myVocabList, postinDoc))

# 贝叶斯分类
p0V, p1V, pAb = bayes.train_NB_0(trainMat, listClasses)
print("贝叶斯分类:\n", p0V,
      "\n", p1V,
      "\n文档属于侮辱类的概率pAb为：", pAb)

# 使用朴素贝叶斯进行交叉验证
bayes.spam_test()

# 使用朴素贝叶斯来发现地域相关的用词
ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('https://sports.yahoo.com/nba/teams/hou/rss.xml')
print("ny`s len = ", len(ny['entries']), ", sf`s len = ", len(sf['entries']))
vocabList, pSF, pNY = bayes.local_words(ny, sf)
# print(vocabList, pSF, pNY)
print(bayes.get_top_words(ny, sf))
