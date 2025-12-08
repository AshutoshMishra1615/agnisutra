import 'package:flutter/material.dart';
import '../services/auth_service.dart';
import '../services/weather_service.dart';
// import '../screens/satellite_map_screen.dart';
import '../screens/select_crop_screen.dart';
import '../screens/add_crop_details_screen.dart';
import '../models/crop_data.dart';
import 'home_carousel.dart';

class DashboardWidget extends StatefulWidget {
  const DashboardWidget({super.key});

  @override
  State<DashboardWidget> createState() => _DashboardWidgetState();
}

class _DashboardWidgetState extends State<DashboardWidget> {
  String userName = "User";
  final _authService = AuthService();
  final _weatherService = WeatherService();
  List<WeatherData> _weatherForecast = [];

  final Map<String, String> _cropIcons = {
    'Sunflower': 'assets/images/icons/Frame 264.png',
    'Mustard': 'assets/images/icons/Frame 265.png',
    'Soyabean': 'assets/images/icons/Frame 266.png',
    'Safflower': 'assets/images/icons/Frame 267.png',
    'Sesame': 'assets/images/icons/Frame 268.png',
    'Niger': 'assets/images/icons/Frame 264 (1).png',
    'Groundnut': 'assets/images/icons/Frame 267 (1).png',
    'Castor': 'assets/images/icons/Frame 267 (2).png',
  };

  final List<CropData> _crops = [
    CropData(
      name: 'Sunflower',
      statusColor: Colors.yellow,
      progress: 45,
      moisture: '3%',
      temp: '28°C',
      sownDate: '24 Nov',
      lastIrrigation: '28 Nov',
      lastPesticide: '29 Nov',
      expectedYield: '25 Dec',
    ),
  ];

  @override
  void initState() {
    super.initState();
    _loadUserName();
    _loadWeather();
  }

  Future<void> _loadUserName() async {
    final name = await _authService.getUserName();
    if (name != null && mounted) {
      setState(() {
        userName = name;
      });
    }
  }

  Future<void> _loadWeather() async {
    final forecast = await _weatherService.getWeeklyForecast(28.6139, 77.2090);
    if (mounted) {
      setState(() {
        _weatherForecast = forecast;
      });
    }
  }

  Future<void> _showAddCropDialog() async {
    final result = await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const SelectCropScreen()),
    );

    if (result == null) return;
    final List<String> selectedCrops = List<String>.from(result);

    if (!mounted) return;

    for (final cropName in selectedCrops) {
      final cropData = await Navigator.push<CropData>(
        context,
        MaterialPageRoute(
          builder: (context) => AddCropDetailsScreen(cropName: cropName),
        ),
      );

      if (cropData != null && mounted) {
        setState(() {
          _crops.add(cropData);
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: const Color(0xFF0D0D0D),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              height: 90,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: _crops.length + 1,
                itemBuilder: (context, index) {
                  if (index == _crops.length) {
                    return _buildAddCropCircle();
                  }
                  return _buildCropCircle(_crops[index]);
                },
              ),
            ),
            const SizedBox(height: 24),
            SizedBox(
              height: 110,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                itemCount: _weatherForecast.length,
                itemBuilder: (context, index) {
                  return _buildWeatherCard(_weatherForecast[index]);
                },
              ),
            ),
            const SizedBox(height: 32),
            const HomeCarousel(),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }

  Widget _buildAddCropCircle() {
    return GestureDetector(
      onTap: _showAddCropDialog,
      child: Padding(
        padding: const EdgeInsets.only(right: 18.0),
        child: Column(
          children: [
            Container(
              width: 60,
              height: 60,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: Colors.white54, width: 1),
              ),
              child: const Icon(Icons.add, color: Colors.white54, size: 30),
            ),
            const SizedBox(height: 10),
            const Text(
              'Add Crop',
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCropCircle(CropData crop) {
    final iconPath = _cropIcons[crop.name];
    return Padding(
      padding: const EdgeInsets.only(right: 16.0),
      child: Column(
        children: [
          Container(
            width: 60,
            height: 60,
            padding: const EdgeInsets.all(0),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              // border: Border.all(color: Colors.amber, width: 2),
            ),
            child: Container(
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                color: Color(0xFF2C2C2C),
              ),
              child: iconPath != null
                  ? ClipOval(
                      child: Image.asset(
                        iconPath,
                        fit: BoxFit.cover,
                        errorBuilder: (context, error, stackTrace) =>
                            const Icon(
                              Icons.local_florist,
                              color: Colors.amber,
                              size: 30,
                            ),
                      ),
                    )
                  : const Icon(
                      Icons.local_florist,
                      color: Colors.amber,
                      size: 30,
                    ),
            ),
          ),
          const SizedBox(height: 10),
          Text(
            crop.name,
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
        ],
      ),
    );
  }

  Widget _buildWeatherCard(WeatherData data) {
    return Container(
      width: 80,
      margin: const EdgeInsets.only(right: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1E1E),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(data.icon, color: Colors.amber, size: 28),
          const SizedBox(height: 8),
          Text(
            '${data.temp}°C',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            data.date,
            style: const TextStyle(color: Colors.white70, fontSize: 10),
          ),
          const SizedBox(height: 2),
          Text(
            data.day,
            style: const TextStyle(color: Colors.white70, fontSize: 10),
          ),
        ],
      ),
    );
  }
}
