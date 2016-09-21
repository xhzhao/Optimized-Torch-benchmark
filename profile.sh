
iter=4
logName='google.txt'
grep Convolution $logName | grep forward | awk '{sum+=$6}END{print "convolution forward 	= " sum/'$iter'}'
grep Convolution $logName | grep bwddata | awk '{sum+=$6}END{print "convolution bwddata 	= " sum/'$iter'}'
grep Convolution $logName | grep bwdfilter | awk '{sum+=$6}END{print "convolution filter 	= " sum/'$iter'}'


grep MaxPooling $logName | grep forward | awk '{sum+=$6}END{print "MaxPooling forward 		= " sum/'$iter'}'
grep MaxPooling $logName | grep backward | awk '{sum+=$6}END{print "MaxPooling backward 	= " sum/'$iter'}'
grep AveragePooling $logName | grep forward | awk '{sum+=$6}END{print "AveragePooling forward 	= " sum/'$iter'}'
grep AveragePooling $logName | grep backward | awk '{sum+=$6}END{print "AveragePooling backward = " sum/'$iter'}'

grep Relu $logName | grep forward | awk '{sum+=$6}END{print "Relu forward 			= " sum/'$iter'}'
grep Relu $logName | grep backward | awk '{sum+=$6}END{print "Relu backward 			= " sum/'$iter'}'

grep LRN $logName | grep forward | awk '{sum+=$6}END{print "LRN forward 			= " sum/'$iter'}'
grep LRN $logName | grep backward | awk '{sum+=$6}END{print "LRN backward 			= " sum/'$iter'}'

grep BatchNorm $logName | grep forward | awk '{sum+=$6}END{print "BatchNorm forward 		= " sum/'$iter'}'
grep BatchNorm $logName | grep backward | awk '{sum+=$6}END{print "BatchNorm backward 		= " sum/'$iter'}'

echo "Linear forward 		= 0"
echo "Linear backward 		= 0"
echo "Dropout forward 		= 0"
echo "Dropout backward 		= 0"
echo "ConcatTable forward 	= 0"
echo "ConcatTable backward 	= 0"

grep Concat $logName | grep forward | awk '{sum+=$6}END{print "Concat forward 		= " sum/'$iter'}'
grep Concat $logName | grep backward | awk '{sum+=$6}END{print "Concat backward 	= " sum/'$iter'}'


echo "Threshold forward 	= 0"
echo "Threshold backward 	= 0"
