lazy val root = (project in file(".")).
  settings(
    name := "wlml",
    version := "0.0.1",
    scalaVersion := "2.11.8",
    libraryDependencies  ++= Seq(
      "org.scalanlp" %% "breeze" % "0.12",
      "org.slf4j" % "slf4j-simple" % "1.7.6"
      //"org.scalanlp" %% "breeze-natives" % "0.12"
    ),
    resolvers ++= Seq(
      "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
    ),
    javaOptions += "-Dcom.github.fommil.netlib.BLAS=com.github.fommil.netlib.NativeRefBLAS"
  )
