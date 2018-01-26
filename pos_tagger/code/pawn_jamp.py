


def pawn_jamp(inputArray,infineLoop,pawn_poistion,total_jamps,index_array):

	if len(inputArray)<=0: 
		return 0
	elif infineLoop:
		return -1
	elif pawn_poistion > len(inputArray):
		return total_jamps
	else:
		pawn_poistion=pawn_poistion+inputArray[pawn_poistion]
		total_jamps+=1
		if pawn_poistion < len(inputArray):
			if index_array[pawn_poistion] <= 3:
				index_array[pawn_poistion]+=1
			else:
				infineLoop = True
		return pawn_jamp(inputArray,infineLoop,pawn_poistion,total_jamps,index_array)


def main():
	inputArray=[2,3,-1,1,3]
	# inputArray=[1,1,-1,1]
	index_array=[0]*len(inputArray)
	print(pawn_jamp(inputArray,False,0,0,index_array))

if __name__ == '__main__':
	main()
