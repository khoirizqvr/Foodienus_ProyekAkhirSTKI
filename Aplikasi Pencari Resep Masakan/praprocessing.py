import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

#Baca data resep masakan dan hapus missing value
data1 = pd.read_csv('dataset-ayam.csv').dropna()
data2 = pd.read_csv('dataset-ikan.csv').dropna()
data3 = pd.read_csv('dataset-kambing.csv').dropna()
data4 = pd.read_csv('dataset-sapi.csv').dropna()
data5 = pd.read_csv('dataset-tahu.csv').dropna()
data6 = pd.read_csv('dataset-telur.csv').dropna()
data7 = pd.read_csv('dataset-tempe.csv').dropna()
data8 = pd.read_csv('dataset-udang.csv').dropna()

# Gabungkan dataset tambahan ke dalam dataset utama (data)
data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8], ignore_index=True)

#Hapus kolom yang tidak diperlukan
data = data.drop(columns='Loves')

#Untuk keperluan output pada aplikasi
data['Ingredientss'] = data['Ingredients'].copy()

# Hapus duplikat 
data.drop_duplicates(subset='Ingredients', keep='first', inplace=True, ignore_index=True)

# Pra-pemrosesan data
def remove_symbols_and_numbers(words):
    clean_words = [re.sub(r'[^a-zA-Z]', ' ', word) for word in words]
    return list(filter(None, clean_words))

# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def apply_stemming(words):
    return [stemmer.stem(word) for word in words]

# Case folding
data['Ingredients'] = data['Ingredients'].str.lower()

# Tokenizing
data['Ingredients'] = data['Ingredients'].apply(lambda x: word_tokenize(str(x)) if type(x) == str else x)

# Remove simbol & angka
data['Ingredients'] = data['Ingredients'].apply(remove_symbols_and_numbers)

# Stopword
custom_stopwords = ['kangkung','pati','egg','patty','micin','salted','apple','sotong','aci','tenggiri','telor','cinamon',
                    'yakiniku','berminyak','sapu','terigu','ketumbar','blanak','gembus','perisa','butternut','cerry','saffron',
                    'gajeh','raisin','pokcay','pisang','spageti','lalap','striploin','fetucini','ladaputih','misoa','chimory','ubi',
                    'kool','edame','itik','mayonise','baksonya','madurasa','rosemary','cheesecake','burung','keudang','oncom','pita',
                    'airnya','nori','jerruk','pastel','garnish','tengkleng','nuts','petis','lengkus','daum','salem','lemusir','mitcin',
                    'berikan','pala','ladaa','kaldunya','sala','santan','cocktail','miso','ayam','iganya','rawonan','kulit','sausnya',
                    'puding','langkoas','nasi','risoles','spagetti','tulangny','ribs','kapur','bijipala','kewpie','kebab','ayamnya',
                    'rayco','mozarela','gulo','mango','sedri','gula','daunya','tortilla','sphagetti','dagingnya','mayonnaise','scramble'
                    ,'udangnya','chocochips','barbeque','sahang','sayapnya','gajihnya','secukupnyagula','santanny','sari','daunn',
                    'terasinya','kecrot','tempenya','polo','kancing','pastr','penyedab','chilli','belimbingnya','ladakku','asapi','koya',
                    'bolognese','mereh','yakisoba','lalapan','palu','sirih','tetelan','fermipan','lemgkuas','keju','bawal','putih',
                    'macarel','gulamerah','selederi','bwgmerh','rib','jinten','karih','kanji','jerok','serreh','cabei','gdaging','pejantan',
                    'hongkong','bokcoy','ajah','vinegar','kale','kabab','pacet','tumisanny','rimpang','spagetty','dumpling','pucung',
                    'mayonais','krosok','marinade','pemyedap','totole','kacang','peyedap','kecap','kwetiau','baberque','sirloin','jogja',
                    'bawangbombai','butih','pipih','kismis','ketmbar','permen','cuko','kempos','tortila','quickmelt','rambak','sao','racik',
                    'daon','chopstick','teriaki','buncis','difillet','opor','baking','suclarose','chedar','balmer','lenjer','cukini','carry',
                    'oreo','rusa','limo','mentega','kuping','pokcoi','gochujiang','mozalla','popcorn','hotdog','lemongrass','nughet','cube',
                    'sushi','ajinamoto','gyoza','kuchai','korned','caplak','suon','kupu','omelette','kecutnya','koktail','serehnya',
                    'sostiram','silky','kentel','gochujang','ice','batter','royico','sarinya','putri','buahjeruk','vanilla','mesis',
                    'ganache','springkle','gurita','mayoneise','inginkan','bungalawang','tongkolnya','oat','keprek','peute','fillet',
                    'organic','digrebek','','cuka','tokolan','saus','padas','cabeijo','gajah','baccon','paku','bangkuang','sapii','kengkeh',
                    'gprek','ketunbar','belulang','sambalnya','pokchoi','banten','bijian','emping','topping','rolade','madu','seladri',
                    'tempet','iceberg','tomatto','litersusu','sawi','hokben','asam','garamnya','korket','pong','kapu','bolognase','eggsnya',
                    'corned','kare','crumps','chilly','winyak','ako','potatoes','timun','fisch','tpungberas','kangkunng','coca','merahnya',
                    'oil','aer','ketuncar','segitigabiru','lunak','bengkuang','water','garam','margarin','kerumbar','petei','seree',
                    'kedondong','merah','mechin','kentucky','toauge','udg','bala','daging','kcambah','tofu','toffu','urat','soysauce',
                    'filled','tipis','rusuk','sempol','apel','sebaguna','oysters','alfamart','rollade','susu','saosnya','natrium','persegi',
                    'suwirny','msg','rosmarry','rendang','crush','kraf','qurban','sun','gulpas','nugget','pasta','yoghurt','nuttela',
                    'cempedak','ikanasin','semur','cangkalan','beefstock','korea','cebe','sardine','cco','asparagus','cambah','fruit',
                    'lundu','meizena','suir','ruasjahe','ekor','nata','onclang','segitiga','aren','air','mami','ceplok','ceker','crime',
                    'jambu','mericak','kecepeh','keladi','kemanggi','ladabubuk','tua','gepeng','macroni','flakes','ajinomoto','uratnya',
                    'agar','cawit','yakitori','spcy','bacem','meiseis','sonice','petai','fresh','shiratake','inggris','pastley','bakar',
                    'bakwan','kue','sisakan','resoles','peanuts','italian','rajang','bango','tempurung','masako','blackpepper','kakapnya',
                    'jaruk','powder','jawa','meses','maezena','bawanng','keripik','zucchini','baso','bihun','lontong','belacan','balimbing',
                    'kaldo','tiram','papper','thailand','serat','pine','labu','ramen','payung','lada','spinach','kluweh','maizena',
                    'gulapasir','lidi','bolognais','indofod','champignons','kates','matasapi','remdang','roti','kambingnya','jengkol',
                    'liquid','bawangputih','seaweed','temukunci','ular','bechamel','wing','soyu','papay','prociz','peper','gulainya',
                    'mayonaise','kedele','sium','penyadap','hyp','kapulogo','cola','sunda','labusiam','pari','juice','crancy','kayu',
                    'berlemak','pecay','muncung','cheri','acar','himalaya','tirem','gajih','ketimbar','wulu','nabati','bajak','mamasuka',
                    'garm','santany','segituga','ginjal','ketupat','sisik','bulu','gaga','india','santang','riyco','ketumbat','sage','kol',
                    'blackpapper','kaleng','cashew','jalapeno','laos','cemangi','kecambahnya','bakmie','maizen','bawaang','soy','kikil',
                    'durinya','berkulit','hershey','proteina','garan','serainya','ampas','oregano','soto','kunir','kuyit','kripik','paprika',
                    'choi','jelantah','leaf','lemon','tempoya','saji','roico','food','carrot','rose','daun','raisins','lengkuas','kupat',
                    'ketimum','telornya','sirup','rawitnya','salmon','garlic','mazena','janten','krmangi','bpt','ulat','green','potatto',
                    'mata','pemanis','lidah','udang','serbaguna','curry','sayur','keprak','gading','lengkkuas','col','bihunnya','daung',
                    'gandis','geprek','cherry','mujahir','salam','wijeen','royco','kfc','ginger','timur','cimory','panir','mangut','pepes',
                    'tenderloin','sledir','taucho','rip','biji','gulmer','dagingngnya','mujaer','puyuh','matah','puyu','gepreg','marie',
                    'alpukat','kambing','gigi','gambas','borokoli','rice','nangka','onion','otak','ceres','kimchi','coklat','gandum','donat',
                    'pockoy','foil','mayumi','soya','kalapa','milk','makaroni','vodka','duri','muffin','nestum','masaco','malkis','probiotik',
                    'yoshi','cabeh','pete','orange','fish','jahe','mayones','santanya','dori','dough','bacang','caos','gembung','krupuk',
                    'bimoli','miri','ketumnar','garram','blinjo','manies','menjamur','kapulaga','peda','mericah','jepang','olive','pasteur',
                    'whiping','lettuce','enoki','patry','gandaria','sumbawa','airdingin','ktumbar','dog','meat','seafood','bluben','cryspy',
                    'urap','cinnamon','tropicana','kerecek','tahunya','breadcrumbs','labuhsiam','cakra','lenjar','rawit','tea','filet','sutra',
                    'lime','sagutani','cardamom','cornet','noodle','juz','dieng','paha','fermifan','bongol','spices','mayonese','sosin',
                    'rocket','sardines','bonggol','gede','kanir','wedges','kepala','rebung','berempah','kira','ginjer','santen','kubis',
                    'kuda','ttomat','macarone','cabenya','worteel','saledri','sumsumnya','moza','kanzi','lobak','kardamon','jerebung',
                    'minyak','limau','kaki','jus','meises','chinese','biskuit','asemnya','kecapnya','peras','secang','tim','naget','timum',
                    'red','oats','puyuhtahu','sardin','pepaya','badan','dipindang','royko','bergajih','baput','ampela','empek','rosmarin',
                    'camba','kari','kayumanis','koepo','mlinjo','arak','cilok','kulitnyaa','fiesta','evaporated','abon','bleder','cream',
                    'pucuknya','shabu','spaghetti','penetralisir','butter','belimbing','parseley','bear','wasabi','sereh','quaker','mentaga',
                    'kraft','makroni','lodeh','crumb','martabak','mayoness','arinya','yamato','sambal','mantega','bbq','serah','krispi',
                    'keluak','buncir','coconut','kamijoro','pizzanya','presto','medoan','bungkul','bawanv','togenya','bulgogi','celedry',
                    'garnise','wajik','dimsum','hijau','kelapanya','ketchup','genjer','caabe','springkel','nasgor','rosbrand','cheese','soun',
                    'telur','kulit','tamagoyaki','spagheti','vegetables','jagung','mozarella','kedongdong','bambu','tapioca','sphageti',
                    'trasi','bangka','kelapa','instant','procis','bonggolnya','angsa','permifan','tiramnya','mangga','ginseng','danging',
                    'rosmery','telurnya','palm','persley','holand','pedas','rawon','kriuk','dadar','fuyunghai','seasalt','pandan','tomato',
                    'petersely','gabus','kecamba','benggol','bandeng','malang','kerapu','almond','chery','cingur','oxtail','gaprek',
                    'tongseng','ketumar','aglio','padang','perkedel','canola','steakny','prancis','hatinya','bokol','leher','greentea',
                    'caisin','pir','tulangnya','jamur','pisangnya','sumsum','putut','kaylan','cekala','roosmery','cabai','mineral',
                    'berasnya','kebuli','marinasi','saltedbutter','kerbau','bokchoy','fettuccini','ladaku','maseko','bangao','wijsman',
                    'fettuccine','lemur','jipang','abc','tebu','selada','steak','sapo','soda','brontak','sajiku','anggur','terung','pere',
                    'sayap','papuyu','ketan','krimer','kakap','perancis','srikaya','broiler','brisket','kurma','pulowaras','batangnya',
                    'lengkuwas','gajinya','keong','cakalang','surawung','marinase','otot','bengkoang','mie','vanila','cabainya','sumedang',
                    'tepungg','chicken','bakpao','marica','choy','coco','tolo','potato','palmia','pizza','risol','dagingny','bunga','donburi',
                    'sawit','vanili','jari','buttermilk','kumisnya','bakung','maisena','jerohan','bijinya','kepiting','indomie','jalar',
                    'melinjo','paper','gabul','curiwis','teri','lencak','kentangnya','kelinci','sengkel','brondolin','kuniran','kerisik',
                    'gulai','origano','siomay','chifferi','blueband','tepungbumbu','gurameh','wagyu','oclang','kristal','rosebrand','terong',
                    'kiwi','cikur','dompol','pickle','boneless','chiken','mandhi','zaitun','buntut','sumatra','magicom','white','scrumble',
                    'benang','barbeqiu','kentaki','cecek','filetan','mawar','champ','tempex','cengkih','reyco','ngohyong','kuas','roast',
                    'lelenya','jeroan','tapioka','sasa','sosis','uduk','kopi','kampung','rempah','plain','saory','rosmary','botok','buah',
                    'maizenna','memanir','frisian','taoco','skm','whipcream','kunyit','daunsalam','ikan','memarinasi','pewarnanya','chees',
                    'disate','youghurt','cireng','sere','margarine','cah','tuge','babat','warna','tepung','hunkwe','tauco','sauce','talas',
                    'serei','tempura','buahnya','es','iwak','tomyam','masyako','tinta','smooked','beri','manis','toge','jelotot','kingfish',
                    'garamm','sirsak','penyedaprasa','pempek','jeruk','peanut','calamarata','mayur','avocado','greenfields','ijo','kornetku',
                    'serre','merrah','kapolaga','kemangii','paru','basmati','himlay','belibis','bumas','mayinaise','daunbawang','bijih',
                    'sausya','sugar','mendoan','ikannya','pangsit','madura','tartar','himalay','spagethi','sodium','jebung','cabemerah',
                    'etawa','tempe','kelor','sledry','mustard','sapi','balado','krecek','pare','pronas','kornet','kapau','mricanya',
                    'dalmonte','pakcoy','dada','tepungnya','sandwich','bolognise','wine','seledry','quacker','pentol','domba','walang',
                    'tahu','cumi','jerbung','mericanya','jintan','kluwek','gendhok','fullcream','sate','masakao','oatmeal','chips','korokeling',
                    'walnut','berry','mashroom','gulanya','cucumber','kemangi','pear','pelezat','lenguas','srundeng','jeruknya','sabu','mrica',
                    'oyster','berdaging','cokelat','nila','teh','saos','tauge','brokolli','shrimps','gurame','rotinya','beras','paniran',
                    'touge','delmonte','mayo','nipis','bawang','royiko','rujak','mayoneis','bamboo','bayam','sup','jantung','serai','paste',
                    'capcay','kalasan','ragi','cookie','belinjo','belut','wartel','spaghetii','palem','somay','siam','mujaher','pecel',
                    'tongkol','jangki','sarimi','tuna','bakso','merica','banana','minyaknya','tempoyak','honey','crab','scallop']
data['Ingredients'] = data['Ingredients'].apply(lambda x: [word for word in x if word.lower() in custom_stopwords])

# Stemming
data['Ingredients'] = data['Ingredients'].apply(apply_stemming)

# Join kata-kata setelah pra-pemrosesan
data['Ingredients'] = data['Ingredients'].apply(lambda x: ' '.join(x))

# Simpan data yang telah diproses menjadi file CSV baru
data.to_csv('data_praproses.csv', index=False)