// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'crop_data.dart';

// **************************************************************************
// TypeAdapterGenerator
// **************************************************************************

class CropDataAdapter extends TypeAdapter<CropData> {
  @override
  final int typeId = 0;

  @override
  CropData read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{
      for (int i = 0; i < numOfFields; i++) reader.readByte(): reader.read(),
    };
    return CropData(
      name: fields[0] as String,
      statusColor: Color(fields[1] as int),
      progress: fields[2] as double,
      moisture: fields[3] as String,
      temp: fields[4] as String,
      sownDate: fields[5] as String,
      lastIrrigation: fields[6] as String,
      lastPesticide: fields[7] as String,
      expectedYield: fields[8] as String,
    );
  }

  @override
  void write(BinaryWriter writer, CropData obj) {
    writer
      ..writeByte(9)
      ..writeByte(0)
      ..write(obj.name)
      ..writeByte(1)
      ..write(obj.statusColorValue)
      ..writeByte(2)
      ..write(obj.progress)
      ..writeByte(3)
      ..write(obj.moisture)
      ..writeByte(4)
      ..write(obj.temp)
      ..writeByte(5)
      ..write(obj.sownDate)
      ..writeByte(6)
      ..write(obj.lastIrrigation)
      ..writeByte(7)
      ..write(obj.lastPesticide)
      ..writeByte(8)
      ..write(obj.expectedYield);
  }

  @override
  int get hashCode => typeId.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is CropDataAdapter &&
          runtimeType == other.runtimeType &&
          typeId == other.typeId;
}
