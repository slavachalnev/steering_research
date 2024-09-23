export interface FeatureData {
	binMax: number;
	binMin: number;
	maxValue: number;
	minValue: number;
	tokens: string[];
	values: number[];
}

export interface Labels {
	labels: string[];
}

export interface ProcessedFeaturesType {
	analysis?: string;
	distillation?: Labels;
	feature: number;
	id: string;
	feature_results: FeatureData[];
}

export interface Activation {
	tokens: string[];
	values: number[];
}
