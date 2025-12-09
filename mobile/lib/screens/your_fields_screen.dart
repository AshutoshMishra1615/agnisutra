import 'package:flutter/material.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:latlong2/latlong.dart';
import 'add_field_screen.dart';
import 'field_details_screen.dart';

class YourFieldsScreen extends StatefulWidget {
  const YourFieldsScreen({super.key});

  @override
  State<YourFieldsScreen> createState() => _YourFieldsScreenState();
}

class _YourFieldsScreenState extends State<YourFieldsScreen> {
  List<Map<String, dynamic>> _fields = [];
  late Box _box;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadFields();
  }

  Future<void> _loadFields() async {
    _box = await Hive.openBox('fields');
    if (_box.isNotEmpty) {
      final data = _box.values.toList();
      setState(() {
        _fields = data.map((e) {
          final map = Map<String, dynamic>.from(e);
          if (map['location'] != null) {
            final loc = Map<String, dynamic>.from(map['location']);
            map['location'] = LatLng(loc['lat'], loc['lon']);
          }
          return map;
        }).toList();
        _isLoading = false;
      });
    } else {
      setState(() {
        _fields = [];
        _isLoading = false;
      });
    }
  }

  Future<void> _saveFields() async {
    final dataToSave = _fields.map((e) {
      final map = Map<String, dynamic>.from(e);
      if (map['location'] != null) {
        final loc = map['location'] as LatLng;
        map['location'] = {'lat': loc.latitude, 'lon': loc.longitude};
      }
      return map;
    }).toList();
    await _box.clear();
    await _box.addAll(dataToSave);
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        backgroundColor: Color(0xFF1E1E1E),
        body: Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Your Fields',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView.builder(
          itemCount: _fields.length + 1,
          itemBuilder: (context, index) {
            if (index == _fields.length) {
              return Container(
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  color: const Color(0xFF386641),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ListTile(
                  title: const Text(
                    'Add Field',
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                  trailing: const Icon(Icons.add, color: Colors.white),
                  onTap: () async {
                    final newFieldName = 'Field ${_fields.length + 1}';
                    final result = await Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) =>
                            AddFieldScreen(fieldName: newFieldName),
                      ),
                    );

                    if (result != null && mounted) {
                      setState(() {
                        _fields.add(result as Map<String, dynamic>);
                        _saveFields();
                      });
                    }
                  },
                ),
              );
            }
            return Container(
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF3E3E3E),
                borderRadius: BorderRadius.circular(12),
              ),
              child: ListTile(
                title: Text(
                  _fields[index]['name'],
                  style: const TextStyle(color: Colors.white, fontSize: 16),
                ),
                trailing: const Icon(
                  Icons.arrow_forward_ios,
                  color: Colors.white,
                  size: 16,
                ),
                onTap: () async {
                  final result = await Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => FieldDetailsScreen(
                        fieldName: _fields[index]['name'],
                        location: _fields[index]['location'],
                      ),
                    ),
                  );

                  if (result != null && mounted) {
                    setState(() {
                      _fields[index] = result as Map<String, dynamic>;
                      _saveFields();
                    });
                  }
                },
              ),
            );
          },
        ),
      ),
    );
  }
}
