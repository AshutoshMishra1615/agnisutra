import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:hive_flutter/hive_flutter.dart';
import '../constants.dart';

class YieldService {
  final Dio _dio = Dio();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  Future<Map<String, dynamic>?> getIoTData() async {
    try {
      String? token = await _storage.read(key: 'access_token');
      if (token == null) {
        throw 'User not authenticated. Please log in again.';
      }
      print('Fetching IoT data from: ${AppConstants.baseUrl}/iot/latest');

      final response = await _dio.get(
        '${AppConstants.baseUrl}/iot/latest',
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );
      if (response.statusCode == 200) {
        return response.data;
      } else {
        print('IoT Data failed with status: ${response.statusCode}');
        throw 'Server returned ${response.statusCode}';
      }
    } catch (e) {
      print('IoT Data error: $e');
      if (e is DioException) {
        if (e.response?.statusCode == 401) {
          await _storage.delete(key: 'access_token');
          throw 'Session expired. Please log in again.';
        }
        if (e.response != null) {
          throw 'IoT Error: ${e.response?.statusCode} - ${e.response?.statusMessage}';
        } else {
          throw 'IoT Connection Error: ${e.message}';
        }
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>?> getWeatherData(double lat, double lon) async {
    try {
      String? token = await _storage.read(key: 'access_token');
      if (token == null) {
        throw 'User not authenticated. Please log in again.';
      }
      final response = await _dio.get(
        '${AppConstants.baseUrl}/krishi-saathi/weather/',
        queryParameters: {'lat': lat, 'lon': lon},
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );
      if (response.statusCode == 200) {
        return response.data;
      }
      throw 'Weather server returned ${response.statusCode}';
    } catch (e) {
      print('Weather Data error: $e');
      if (e is DioException) {
        print('Failed URL: ${e.requestOptions.uri}');
        if (e.response?.statusCode == 401) {
          await _storage.delete(key: 'access_token');
          throw 'Session expired. Please log in again.';
        }
        if (e.response != null) {
          throw 'Weather Error: ${e.response?.statusCode}';
        } else {
          throw 'Weather Connection Error: ${e.message}';
        }
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>?> getYieldPrediction({
    required double nitrogen,
    required double phosphorus,
    required double potassium,
    required double temperature,
    required double humidity,
    required double rainfall,
    required String crop,
  }) async {
    try {
      String? token = await _storage.read(key: 'access_token');
      if (token == null) {
        throw 'User not authenticated. Please log in again.';
      }
      final response = await _dio.get(
        '${AppConstants.baseUrl}/krishi-saathi/get-yield-prediction',
        queryParameters: {
          'nitrogen': nitrogen,
          'phosphorus': phosphorus,
          'potassium': potassium,
          'temperature': temperature,
          'humidity': humidity,
          'ph': 6.5,
          'rainfall': rainfall,
          'crop': crop.toLowerCase(),
        },
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        return response.data;
      }
      throw 'Prediction server returned ${response.statusCode}';
    } catch (e) {
      print('Yield Prediction error: $e');
      if (e is DioException) {
        print('Failed URL: ${e.requestOptions.uri}');
        if (e.response != null) {
          throw 'Prediction Error: ${e.response?.statusCode}';
        } else {
          throw 'Prediction Connection Error: ${e.message}';
        }
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>?> getFertilizerRecommendation({
    required String crop,
    required double targetYield,
    required double soilN,
    required double soilP,
    required double soilK,
    required double temperature,
    required double moisture,
    double ph = 6.5,
  }) async {
    try {
      String? token = await _storage.read(key: 'access_token');
      if (token == null) {
        throw 'User not authenticated. Please log in again.';
      }
      final response = await _dio.post(
        '${AppConstants.baseUrl}/krishi-saathi/recommend',
        data: {
          'crop': crop,
          'target_yield': targetYield,
          'soil_N': soilN,
          'soil_P': soilP,
          'soil_K': soilK,
          'temperature': temperature,
          'ph': ph,
          'moisture': moisture,
        },
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        // Cache the recommendation for AI context
        var box = await Hive.openBox('last_recommendation');
        await box.put('data', response.data);
        return response.data;
      }
      throw 'Recommendation server returned ${response.statusCode}';
    } catch (e) {
      print('Recommendation error: $e');
      if (e is DioException) {
        if (e.response?.statusCode == 401) {
          await _storage.delete(key: 'access_token');
          throw 'Session expired. Please log in again.';
        }
        if (e.response != null) {
          throw 'Recommendation Error: ${e.response?.statusCode}';
        } else {
          throw 'Recommendation Connection Error: ${e.message}';
        }
      }
      rethrow;
    }
  }

  Future<Map<String, dynamic>?> getPrediction(
    Map<String, dynamic> formData,
  ) async {
    var box = await Hive.openBox('last_prediction');

    try {
      String? token = await _storage.read(key: 'access_token');

      final response = await _dio.post(
        '${AppConstants.baseUrl}/predict',
        data: formData,
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );

      if (response.statusCode == 200) {
        // Save to Hive
        await box.put('data', response.data);
        await box.put('is_offline', false);
        return response.data;
      }
      return null;
    } catch (e) {
      print('Prediction error: $e');
      // Load from Hive if available
      if (box.containsKey('data')) {
        final data = Map<String, dynamic>.from(box.get('data'));
        data['is_offline'] = true; // Flag to show UI
        return data;
      }
      return null;
    }
  }

  Future<Map<String, dynamic>?> getNDVI(double lat, double lon) async {
    try {
      String? token = await _storage.read(key: 'access_token');
      if (token == null) {
        throw 'User not authenticated. Please log in again.';
      }
      final response = await _dio.get(
        '${AppConstants.baseUrl}/krishi-saathi/ndvi',
        queryParameters: {'lat': lat, 'lon': lon},
        options: Options(headers: {'Authorization': 'Bearer $token'}),
      );
      if (response.statusCode == 200) {
        return response.data as Map<String, dynamic>;
      }
      throw 'NDVI server returned ${response.statusCode}';
    } catch (e) {
      print('NDVI Data error: $e');
      if (e is DioException) {
        if (e.response?.statusCode == 401) {
          await _storage.delete(key: 'access_token');
          throw 'Session expired. Please log in again.';
        }
        if (e.response != null) {
          throw 'NDVI Error: ${e.response?.statusCode}';
        } else {
          throw 'NDVI Connection Error: ${e.message}';
        }
      }
      rethrow;
    }
  }
}
