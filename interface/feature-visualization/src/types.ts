export interface FeatureData {
	binMax: number;
	binMin: number;
	maxValue: number;
	minValue: number;
	tokens: string[];
	values: number[];
}

export interface ProcessedFeaturesType {
	feature: number;
	id: string;
	feature_results: FeatureData[];
}

export interface Activation {
	tokens: string[];
	values: number[];
}
