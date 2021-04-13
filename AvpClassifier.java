package aicertifai.training.classification;

//import necessary libraries

public class AvPClassifier {
	private static String[] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
	
	// predefined the var that will be used
	private static double trainPerc = 0.8;
	public static Random rng = new Random();
	private static PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
	private static int width = 80;
	private static int height = 80;
	private static int nChannels = 3;
	private static int batchSize = 20;
	private static double lr = 1e-3;
	private static int nEpoch = 15;
	private static int seed = 51;
	private static double regTerm = 0.0001;
	private static int nClasses = 3;

	public static void main(String[] args) throws IOException {

		File inputFile = new ClassPathResource("AvP").getFile();

		FileSplit split = new FileSplit(inputFile, allowedExt);

		// used to navigate which folder to read the datasets
		// random path filter vs balanced path filter: random pf are taking random samples from each class , while balanced pf has an arrangement between classes
		// usually used balanced path filter in case of imbalance class

		PathFilter pathFilter = new BalancedPathFilter(rng, allowedExt, labelMaker);

		// train and test splitting
		InputSplit [] allData = split.sample(pathFilter, trainPerc, 1-trainPerc);
		InputSplit trainData = allData[0];
		InputSplit testData = allData[1];

		// Image Augmentation

		ImageTransform hFlip = new FlipImageTransform(0); // 0 - horizontal, 1 - vertical
		ImageTransform rotate = new RotateImageTransform(15);
		RandomCropTransform rCrop = new RandomCropTransform(60, 60);

		List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
			new Pair<>(hFlip, 0.2),
			new Pair<>(rotate, 0.3),
			new Pair<>rCrop, 0.2)
		);

		PipelineImageTransform transform = new PipeLineImageTransform(pipeline, false);

		// load image data

		RecordReader trainRR = new ImageRecordReader(height, width, nChannels, labelMaker);
		RecordReader testRR = new ImageRecordReader(height, width, nChannels, labelMaker);

		trainRR.initialize(trainData, transform);	// apply in train dataset only
		testRR.initialize(testData);

		
		DataSetIterator trainIter = new ImageRecordReaderDataSetIterator(trainRR, batchSize, 1, nClasses);	// label image index is always 1
		DataSetIterator testIter = new ImageRecordReaderDataSetIterator(testRR, batchSize, 1, nClasses);

		// feature scaling - data normalization

		DataNormalization scaler = new ImagePreProcessingScaler();
		trainIter.setPreProcessor(scaler);
		testIter.setPreProcessor(scaler);

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.weightInit(WeightInit.XAVIER)
			.updater(new Adam(lr))
			.seed(seed)
			.list()
			.layer(new ConvolutionLayer.Builder()
				.nIn(nChannels)
				.nOut(16)			// filter count
				.activation(Activation.RELU)
				.kernelSize(3,3) 	//controls how big or small the size is
				.stride(1,1)		//controls the movement
				.build())
			//pooling layer
			.layer(new SubSamplingLayer.Builder()
				.kernelSize(2,2)
				.stride(2,2)
				.poolingType(SubSamplingLayer.PoolingType.MAX)
				.build())
			//fully connected layer
			.layer(new DenseLayer.Builder()
				.activation(Activation.RELU)
				.nOut(20)
				.build())
			//
			.layer(new OutputLayer.Builder()
				.nOut(nClasses)
				.activation(Activation.SOFTMAX)
				.lossFunction(LossFunctions.LossFunction.NEGATIVELOHLIKELIHOOD)
				.build())
			.setInputType.convolutional(height, weight, nChannels))
			.build();

		//model initialization
		MultiLayerNetwork model = new MulriLayerNetwork(conf);
		model.init();

		//model training and set score listeners
		model.setListeners(new ScoreIterationListener(10));
		model.fit(trainIter, nEpoch);

		//model evaluation
		Evaluation evalTrain = model.evaluate(trainIter);
		Evaluation evalTest = model.evaluate(testIter);

		System.out.println("Train Evaluation:\n " + evalTrain.stats());
		System.out.println("Train Evaluation:\n " + evalTrain.stats());

	}
}