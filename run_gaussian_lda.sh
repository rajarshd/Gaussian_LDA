usage() { echo "Usage: $0 [-n <num_iterations> -c <corpus file> -i <input vector file> -d  <num_dimensions> -k <num_topics> " 1>&2;
	exit 1; }

SAMPLE_APHA=0 #by default donot sample for alpha

	while getopts ":d:i:n:k:v:c:a:" o; do
		case "${o}" in
			d)
				NUM_DIM=$OPTARG
				;;
			c)
				CORPUS_FILE=$OPTARG
				;;
			i) 
				INPUT_FILE=$OPTARG
				;;
			n)
				NUM_ITER=$OPTARG
				;;
			k)
				INITIAL_K=$OPTARG
				;;
			\?)
				echo "Invalid option -$OPTARG"
				exit 1;
				;;
			:)
				echo "Option -$OPTARG requires an argument." >&2
				exit 1;
				;;
		esac
	done

if [ -z $NUM_DIM ];
then
	usage;
	exit 1;
fi
if [ -z $INPUT_FILE ];
then
	usage;
	exit 1;
fi
if [ -z $CORPUS_FILE ];
then
	usage;
	exit 1;
fi
if [ -z $NUM_ITER ];
then
	usage;
	exit 1;
fi

#Create the directory structure
time_stamp=`date "+%b_%d_%Y_%H.%M.%S"` 
DIR_NAME='output'/'D'$NUM_DIM/'I'$NUM_ITER/'K'$INITIAL_K/'GLDA'/$time_stamp/
mkdir -p $DIR_NAME
echo 'The output directory is' $DIR_NAME
#Delete the bin directory first so that it doesnot run the old version when compilation breaks
rm -rf bin/*

#compile
javac -sourcepath src/ -d bin/ -cp "external_libs/ejml-0.25.jar:external_libs/commons-logging-1.2/commons-logging-1.2.jar:external_libs/commons-math3-3.3/commons-math3-3.3.jar" src/sampler/GaussianLDA.java

#run
java  -Xmx70g  -cp "bin/:external_libs/ejml-0.25.jar:external_libs/commons-logging-1.2/commons-logging-1.2.jar:external_libs/commons-math3-3.3/commons-math3-3.3.jar" sampler/GaussianLDA $INPUT_FILE $NUM_DIM $NUM_ITER $INITIAL_K $DIR_NAME $CORPUS_FILE
