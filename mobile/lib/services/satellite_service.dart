import 'dart:convert';
import 'package:http/http.dart' as http;
import '../constants.dart';

class SatelliteService {
  final String baseUrl = AppConstants.baseUrl;

  Future<Map<String, dynamic>> getSatelliteData(double lat, double lon) async {
    try {
      print(
        'Fetching satellite data from: $baseUrl/krishi/get-ndvi?lat=$lat&lon=$lon',
      );
      // Your Backend should have an endpoint that returns AgroMonitoring data
      final response = await http
          .get(Uri.parse('$baseUrl/krishi/get-ndvi?lat=$lat&lon=$lon'))
          .timeout(const Duration(seconds: 3));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception(
          'Failed to load satellite data: ${response.statusCode}',
        );
      }
    } catch (e) {
      print('Satellite Service Error: $e. Using Fallback Data.');
      // Fallback Data for Hackathon Stability
      return {
        "ndvi_value": 0.75,
        "image_url": null, // No overlay in offline mode
        "bounds": null,
      };
    }
  }
}
