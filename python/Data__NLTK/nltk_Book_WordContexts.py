import nltk
#nltk.download() #need to download these in specific paths that the tool will look for
from nltk.book import *

# .concordance() - looks for a word and permits us to see it in context; from what I can tell the tool saves these contexts
# .similar() - looks for a similar word; from what I can observer the tool leverages the contexts mined from the concordance()
# .common_context() - seems to be looking for intersections of contexts on 2 different list of contexts from the concordance()

text1.concordance("monstrous")
'''
Displaying 11 of 11 matches:
ong the former , one was of a most monstrous size . ... This came towards us , 
ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r
ll over with a heathenish array of monstrous clubs and spears . Some were thick
d as you gazed , and wondered what monstrous cannibal and savage could ever hav
that has survived the flood ; most monstrous and most mountainous ! That Himmal
they might scout at Moby Dick as a monstrous fable , or still worse and more de
th of Radney .'" CHAPTER 55 Of the Monstrous Pictures of Whales . I shall ere l
ing Scenes . In connexion with the monstrous pictures of whales , I am strongly
ere to enter upon those still more monstrous stories of them which are to be fo
ght have been rummaged out of this monstrous cabinet there is no telling . But 
of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u
'''

text1.concordance("perilous")
'''
Displaying 15 of 15 matches:
s were on the start ; that one most perilous and long voyage ended , only begins
tting out in their canoes , after a perilous passage they discovered the island 
lf , as I myself , hast seen many a perilous time ; thou knowest , Peleg , what 
 good , a ship bound on so long and perilous a voyage -- beyond both stormy Cape
often evinced by others in the more perilous vicissitudes of the fishery . " I w
the forward part of the ship by the perilous seas that burstingly broke over its
All the oarsmen are involved in its perilous contortions ; so that to the timid 
after much weary pulling , and many perilous , unsuccessful onsets , he at last 
und his waist . It was a humorously perilous business for both of us . For , bef
ly revealing by those struggles the perilous depth to which he had sunk . At thi
march , all eagerness to place that perilous passage in their rear , and once mo
yet does it present one of the more perilous vicissitudes of the fishery . For a
. For example ,-- after a weary and perilous chase and capture of a whale , the 
 shore is intended to carry off the perilous fluid into the soil ; so the kindre
avenly quadrant ? and in these same perilous seas , gropes he not his way by mer
'''

text1.similar("monstrous")
'''
true contemptible christian abundant few part mean careful puzzled
mystifying passing curious loving wise doleful gamesome singular
delightfully perilous fearless
'''
text1.similar("perilous")
'''
by the in now with of take what that not for more have long find
therefore at as here thought
'''

text1.common_contexts(["monstrous"])
'''
most_size that_bulk of_clubs what_cannibal most_and a_fable
the_pictures more_stories this_cabinet a_size
'''
text1.common_contexts(["perilous"])
'''
most_and a_passage a_time and_a more_vicissitudes the_seas
its_contortions many_unsuccessful humorously_business the_depth
that_passage and_chase the_fluid same_seas
'''
text1.common_contexts(["monstrous", "perilous"])
'''
most_and
'''