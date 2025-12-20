import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import '../services/yield_service.dart';
import 'add_field_screen.dart';

class FieldDetailsScreen extends StatefulWidget {
  final String fieldName;
  final LatLng? location;

  const FieldDetailsScreen({super.key, required this.fieldName, this.location});

  @override
  State<FieldDetailsScreen> createState() => _FieldDetailsScreenState();
}

class _FieldDetailsScreenState extends State<FieldDetailsScreen> {
  final YieldService _yieldService = YieldService();
  String _ndviValue = "Loading...";
  String _ndviPeak = "Loading...";
  String _ndviFinal = "Loading...";
  bool _isLoading = true;
  String? _error;
  String _moistureValue = "Loading...";
  bool _isLoadingMoisture = true;

  @override
  void initState() {
    super.initState();
    _fetchNDVI();
    _fetchIoTData();
  }

  Future<void> _fetchIoTData() async {
    try {
      final iotData = await _yieldService.getIoTData();
      if (iotData != null && mounted) {
        setState(() {
          _moistureValue = (iotData['moisture'] ?? 'N/A').toString();
          _isLoadingMoisture = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _moistureValue = "N/A";
          _isLoadingMoisture = false;
        });
      }
    }
  }

  Future<void> _fetchNDVI() async {
    if (widget.location == null) {
      setState(() {
        _ndviValue = "N/A";
        _ndviPeak = "N/A";
        _ndviFinal = "N/A";
        _isLoading = false;
      });
      return;
    }

    try {
      final ndviData = await _yieldService.getNDVI(
        widget.location!.latitude,
        widget.location!.longitude,
      );
      setState(() {
        if (ndviData != null) {
          _ndviValue = (ndviData['ndvi_flowering'] ?? ndviData['mean'] ?? 'N/A')
              .toString();
          _ndviPeak = (ndviData['ndvi_peak'] ?? ndviData['max'] ?? 'N/A').toString();
          _ndviFinal = (ndviData['ndvi_veg_slope'] ?? ndviData['final'] ?? 'N/A')
              .toString();
        } else {
          _ndviValue = "N/A";
          _ndviPeak = "N/A";
          _ndviFinal = "N/A";
        }
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: Text(
          widget.fieldName,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // NDVI Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF3E3E3E),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(
                        Icons.satellite_alt,
                        color: Colors.white70,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'Satellite Health (NDVI)',
                        style: TextStyle(color: Colors.white70, fontSize: 14),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Text(
                        _isLoading ? "..." : "$_ndviValue %",
                        style: const TextStyle(
                          color: Color(0xFF00E676),
                          fontSize: 32,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFF00E676).withOpacity(0.2),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: const Color(0xFF00E676)),
                        ),
                        child: const Text(
                          'Healthy',
                          style: TextStyle(
                            color: Color(0xFF00E676),
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Peak NDVI',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _isLoading ? "..." : _ndviPeak,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Final NDVI',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _isLoading ? "..." : _ndviFinal,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: const BoxDecoration(
                          color: Color(0xFF00E676),
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'Source : Live Satellite Feed',
                        style: TextStyle(color: Colors.white70, fontSize: 12),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'NDVI values range from -1 to +1. Values above 0.5 indicate dense, healthy vegetation. Low values may suggest water stress or disease.',
                    style: TextStyle(color: Colors.white60, fontSize: 12),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Moisture Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFF3E3E3E),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(
                        Icons.sensors,
                        color: Colors.white70,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'Moisture Detected from Sensor',
                        style: TextStyle(color: Colors.white70, fontSize: 14),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Text(
                        _isLoadingMoisture ? "..." : "$_moistureValue %",
                        style: const TextStyle(
                          color: Color(0xFF00E676),
                          fontSize: 32,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFF00E676).withOpacity(0.2),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: const Color(0xFF00E676)),
                        ),
                        child: const Text(
                          'Good Moisture',
                          style: TextStyle(
                            color: Color(0xFF00E676),
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Map
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.grey[700],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: widget.location == null
                    ? const Center(
                        child: Padding(
                          padding: EdgeInsets.all(16.0),
                          child: Text(
                            "Location not available. Please create a new crop assigned to a field with valid location.",
                            style: TextStyle(color: Colors.white),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: FlutterMap(
                          options: MapOptions(
                            initialCenter: widget.location!,
                            initialZoom: 15.0,
                          ),
                          children: [
                            TileLayer(
                              urlTemplate:
                                  'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                              userAgentPackageName: 'com.example.mobile',
                            ),
                            MarkerLayer(
                              markers: [
                                Marker(
                                  point: widget.location!,
                                  width: 40,
                                  height: 40,
                                  child: const Icon(
                                    Icons.location_on,
                                    color: Colors.red,
                                    size: 40,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
              ),
            ),

            const SizedBox(height: 16),

            // Buttons
            Row(
              children: [
                Expanded(
                  child: OutlinedButton(
                    onPressed: () async {
                      final result = await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => AddFieldScreen(
                            fieldName: widget.fieldName,
                            existingLocation: widget.location,
                          ),
                        ),
                      );
                      // If we get a result back (updated field), we should probably update the UI.
                      // Since this screen takes props, we might need to reload or pop back.
                      // For now, let's just pop back to YourFieldsScreen with the result if it exists,
                      // so YourFieldsScreen can update its list.
                      if (result != null && context.mounted) {
                        Navigator.pop(context, result);
                      }
                    },
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Color(0xFFEF5350)),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(24),
                      ),
                    ),
                    child: const Text(
                      'Edit Field',
                      style: TextStyle(color: Color(0xFFEF5350), fontSize: 16),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton(
                    onPressed: () async {
                      final confirm = await showDialog<bool>(
                        context: context,
                        builder: (context) => AlertDialog(
                          backgroundColor: const Color(0xFF3E3E3E),
                          title: const Text(
                            "Delete Field",
                            style: TextStyle(color: Colors.white),
                          ),
                          content: const Text(
                            "Are you sure you want to delete this field? This action cannot be undone.",
                            style: TextStyle(color: Colors.white70),
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(context, false),
                              child: const Text("Cancel"),
                            ),
                            TextButton(
                              onPressed: () => Navigator.pop(context, true),
                              child: const Text(
                                "Delete",
                                style: TextStyle(color: Colors.redAccent),
                              ),
                            ),
                          ],
                        ),
                      );

                      if (confirm == true && context.mounted) {
                        Navigator.pop(context, {'delete': true});
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFFEF5350),
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(24),
                      ),
                    ),
                    child: const Text(
                      'Delete Field',
                      style: TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
